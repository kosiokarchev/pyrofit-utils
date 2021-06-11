from __future__ import annotations

from io import BytesIO
from itertools import chain, product
from typing import Iterable, Union

import torch
import torch.nn.functional
from more_itertools import last, split_into
from packaging import version


def map_location(obj, location=None):
    """Brute-force remap locations by round-trip serialisation."""
    buf = BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, map_location=location)


def broadcast_except(*tensors: torch.Tensor, dim=-1):
    shape = torch.broadcast_tensors(*[t.select(dim, 0) for t in tensors])[0].shape
    return [t.expand(*shape[:t.ndim + dim + 1], t.shape[dim], *shape[t.ndim + dim + 1:])
            for t in pad_dims(*tensors, ndim=len(shape)+1)]


if version.parse(torch.__version__) < version.parse('1.8'):
    def broadcast_shapes(*shapes):
        with torch.no_grad():
            scalar = torch.zeros((), device="cpu")
            return torch.broadcast_tensors(*(scalar.expand(shape) for shape in shapes))[0].shape
else:
    broadcast_shapes = torch.broadcast_shapes


def broadcast_left(*tensors, ndim):
    shape = broadcast_shapes(*(t.shape[:ndim] for t in tensors))
    return (t.expand(*shape, *t.shape[ndim:]) for t in tensors)


def broadcast_gather(input, dim, index, sparse_grad=False, index_ndim=1):
    """
    input: Size(batch_shape..., N, event_shape...)
    index: Size(batch_shape..., M)
        -> Size(batch_shape..., M, event_shape...)
    """
    index_shape = index.shape[-index_ndim:]
    index = index.flatten(-index_ndim)
    batch_shape = broadcast_shapes(input.shape[:dim], index.shape[:-1])
    input = input.expand(*batch_shape, *input.shape[dim:])
    index = index.expand(*batch_shape, index.shape[-1])
    return torch.gather(input, dim, index.reshape(
        *index.shape, *(input.ndim - index.ndim)*(1,)).expand(
        *index.shape, *input.shape[index.ndim:]
    ) if input.ndim > index.ndim else index, sparse_grad=sparse_grad).reshape(*index.shape[:-1], *index_shape, *input.shape[dim % len(input.shape) + 1:])


# TODO: improve so that nbatch=-1 means "auto-derive nbatch from number of
#  matching dimensions on the left"
def pad_dims(*tensors: torch.Tensor, ndim: int = None, nbatch: int = 0) -> list[torch.Tensor]:
    """Pad shapes with ones on the left until at least `ndim` dimensions."""
    if ndim is None:
        ndim = max([t.ndim for t in tensors])
    return [t.reshape(t.shape[:nbatch] + (1,)*(ndim-t.ndim) + t.shape[nbatch:]) for t in tensors]


def align_dims(t: torch.Tensor, ndims: Iterable[int], target_ndims: Iterable[int]):
    assert sum(ndims) == t.ndim
    return t.reshape(*chain.from_iterable(
        (target_ndim - len(s)) * [1] + s for s, target_ndim
        in zip(split_into(t.shape, ndims), target_ndims)
    ))


def aligned_expand(t: torch.Tensor, ndims: Iterable[int], shapes: Iterable[torch.Size]):
    return align_dims(t, ndims, map(len, shapes)).expand(*chain.from_iterable(shapes))


def num_to_tensor(*args, device=None):
    return [torch.as_tensor(a, dtype=torch.get_default_dtype(), device=device) if not torch.is_tensor(a) else a.to(dtype=torch.get_default_dtype(), device=device)
            for a in args]
    # return [a.to(device) if torch.is_tensor(a) else torch.tensor(a, device=device) for a in args]


def onehotnd(p: torch.tensor, ranges: torch.Size):
    ndim = p.shape[-1]
    onehot = torch.zeros(p.shape[:-2] + ranges, device=p.device)

    onehot_flat = onehot.reshape((-1,) + onehot.shape[-ndim:])
    p_flat = p.reshape((-1,) + p.shape[-2:])
    index = p_flat.long()

    offsets = torch.tensor(list(product(*([[0, 1]] * ndim))),
                           device=p.device)

    for offset in offsets:
        i = index + offset
        onehot_flat.index_put_(((torch.arange(onehot_flat.shape[0], device=p.device)[:, None],)
                                + tuple(i.permute(-1, 0, 1))),
                               (1. - (p_flat - i).abs()).prod(-1),
                               accumulate=True)

    return onehot


def _diff_one(a: torch.Tensor, axis: int):
    return torch.narrow(a, axis, 1, a.shape[axis]-1) - torch.narrow(a, axis, 0, a.shape[axis]-1)


def _mid_many(a: torch.Tensor, axes: Iterable[int]) -> torch.Tensor:
    axes = [ax % a.ndim for ax in axes]
    return last(
        _a for _a in [a] for ax in axes
        for _a in [torch.narrow(_a, ax, 0, _a.shape[ax]-1) + torch.narrow(_a, ax, 1, _a.shape[ax]-1)]
    ) / 2**len(axes) if axes else a


def gradient(a: torch.Tensor, axis: Union[int, Iterable[int]]):
    return (_diff_one(a, axis) if isinstance(axis, int) else
            [_mid_many(_diff_one(a, i), set(axis) - {i}) for i in axis])


def unravel_index(indices: torch.LongTensor, shape: torch.Size) -> torch.LongTensor:
    strides = torch.tensor([p for p in [1] for s in shape[:0:-1] for p in [s*p]][::-1] + [1]).to(indices)
    shape = torch.tensor(list(shape)).to(indices)
    return (indices.unsqueeze(-1) // strides) % shape


class ConvNDFFT:
    def __init__(self, kernel: torch.tensor, ndim: int):
        self.ndim = ndim
        self.kernel = kernel.roll((torch.tensor(kernel.shape[-ndim:]) // 2).tolist(),
                                  list(range(-ndim, 0)))
        self.kernel_fft = torch.rfft(self.kernel, self.ndim, onesided=False)

    def __call__(self, signal, sumdims=None):
        torch.cuda.empty_cache()
        signal_fft = torch.rfft(signal, self.ndim, onesided=False)
        res = torch.empty_like(torch.broadcast_tensors(signal_fft, self.kernel_fft)[0])
        res[..., 0] = (signal_fft[..., 0] * self.kernel_fft[..., 0]
                       - signal_fft[..., 1] * self.kernel_fft[..., 1])
        res[..., 1] = (signal_fft[..., 0] * self.kernel_fft[..., 1]
                       + signal_fft[..., 1] * self.kernel_fft[..., 0])
        res = torch.irfft(res, self.ndim, onesided=False)

        return res if sumdims is None else res.sum(sumdims)


class TorchInterpNd:
    """Curently only works in 2D and 3D because of limitations in torch's grid_sample"""
    def __init__(self, data, *ranges):
        self.ranges = torch.tensor(ranges, dtype=torch.get_default_dtype(), device=data.device)
        self.extents = self.ranges[:, 1] - self.ranges[:, 0]
        self.ndim = len(ranges)

        self.data = data.unsqueeze(0) if data.ndim == self.ndim else data
        assert self.data.ndim == self.ndim + 1
        self.channels = self.data.shape[0]

    def __call__(self, *p_or_args):
        p = p_or_args if len(p_or_args) == 1 else torch.stack(torch.broadcast_tensors(*p_or_args), -1)
        assert p.shape[-1] == self.ndim

        p = 2 * (p - self.ranges[:, 0]) / self.extents - 1

        p_flat = p.reshape(*((1,) * self.ndim), -1, self.ndim)
        data_flat = self.data.unsqueeze(0)

        res = torch.nn.functional.grid_sample(data_flat, p_flat, align_corners=True)
        return torch.movedim(res.reshape(self.channels, *p.shape[:-1]), 0, -1)
