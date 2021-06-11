import typing
from warnings import warn

import pyro.distributions
import torch

try:
    from torchinterp1d import Interp1d as _Interp1d

    # Allow pickling of Interp1d
    # TODO: Check if this is still relevant
    class Interp1d(_Interp1d):
        def __getstate__(self):
            return {}

        def __call__(self, x, y, xnew, out=None):
            return super().__call__(*(_.view(-1, _.shape[-1]) for _ in (x, y, xnew)), out=out).view(xnew.shape)

    interp1d = Interp1d()

except ImportError:
    warn('torchinterp1d not found', category=ImportWarning)
    pass


_size = typing.Union[torch.Size, typing.List[int], typing.Tuple[int, ...]]
_Distribution = typing.Union[
    pyro.distributions.TorchDistribution,
    pyro.distributions.torch_distribution.TorchDistributionMixin,
]
_Tensor_like = typing.Union[torch.Tensor, float]


def unwrap_distribution(dist):
    while hasattr(dist, "base_dist"):
        dist = getattr(dist, "base_dist")
    return dist


def broadcastable(*args: _size):
    return all(len(set(ss)) <= (2 if 1 in ss else 1) for ss in zip(*args))


def broadcast_shapes(*args: torch.Tensor, check=False) -> torch.Size:
    shapes = [p.shape for p in args]
    ndim = max([len(sh) for sh in shapes], default=0)
    shapes = [(1,) * (ndim - len(sh)) + sh for sh in shapes]
    if check:
        assert broadcastable(*shapes)
    return torch.Size(max(s) for s in zip(*shapes))
