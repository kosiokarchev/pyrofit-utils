import typing

import pyro
import torch
from torch._C import _disabled_torch_function_impl
from pyro.distributions import TorchDistribution

from .._utils import _size, broadcastable, broadcast_shapes, Interp1d


class TransformedSample(torch.Tensor):
    __torch_function__ = _disabled_torch_function_impl

    def __new__(cls, data, orig_sample, *args, **kwargs):
        return torch.Tensor._make_subclass(cls, data, *args, **kwargs)

    def __init__(self, data: torch.Tensor, orig_sample: torch.Tensor):
        # data parameter is needed to match __new__ signature
        super().__init__()
        self.orig_sample = orig_sample

    def reshape(self, shape: _size) -> "TransformedSample":
        return type(self)(super().reshape(shape), self.orig_sample.reshape(shape))

    def permute(self, dims: _size) -> "TransformedSample":
        return type(self)(super().permute(dims), self.orig_sample.permute(dims))


class PDF:
    @property
    def param_ndim(self):
        return len(self.param_shape)

    def __init__(self, *params: torch.Tensor, func: typing.Optional[typing.Callable] = None):
        self.param_shape = broadcast_shapes(*params)
        self.func = func

    def __call__(self, *args, **kwargs):
        if self.func is None:
            raise NotImplementedError
        else:
            return self.func(*args, **kwargs)

    wrap = classmethod(lambda cls, func: cls(func=func))


class InverseTransformDistribution(TorchDistribution):
    @property
    def arg_constraints(self):
        return {}

    has_rsample = True

    def __init__(
        self,
        log_prob: typing.Union[PDF, typing.Callable[[torch.Tensor], torch.Tensor]],
        grid: torch.Tensor,
        expand_grid=True,
        log_prob_of_original=False,
    ):
        self._log_prob = log_prob if isinstance(log_prob, PDF) else PDF(func=log_prob)

        if expand_grid:
            grid = grid.reshape(*grid.shape, *(1,) * self._log_prob.param_ndim)
        else:
            assert broadcastable(grid.shape, self._log_prob.param_shape)

        p = log_prob(grid).exp()
        cdf = torch.cat(
            (
                torch.zeros_like(p[:1]),
                torch.cumsum((grid[1:] - grid[:-1]) * (p[1:] + p[:-1]) / 2, dim=0),
            ),
            dim=0,
        )
        norm = cdf[-1]
        cdf = cdf / norm

        self.x, self.y = (
            (cdf, grid)
            if cdf.ndim == 1
            else (
                cdf.flatten(start_dim=1).t(),
                grid.expand_as(cdf).flatten(start_dim=1).t(),
            )
        )

        super().__init__(batch_shape=cdf.shape[1:], event_shape=torch.Size([]))

        self.log_scale = torch.log(norm)
        self._support = pyro.distributions.constraints.interval(grid[0], grid[-1])
        self.sampling_distribution = self.make_sampling_distribution(
            dtype=self.x.dtype, device=self.x.device
        )
        self.log_prob_of_original = log_prob_of_original
        self.interp1d = Interp1d()

    @staticmethod
    def make_sampling_distribution(dtype, device) -> torch.distributions.Distribution:
        return torch.distributions.Uniform(
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
        )

    @property
    def support(self):
        return self._support

    @property
    def mean(self):
        x = (self.y[..., 1:] + self.y[..., :-1]) / 2.0
        dx = self.y[..., 1:] - self.y[..., :-1]
        prob = self._log_prob(x).exp() * dx
        return ((x * prob).sum(-1) / prob.sum(-1)).reshape(self.batch_shape)

    def cdf(self, y: torch.Tensor) -> torch.Tensor:
        nextradim = len(y.shape) - len(self.batch_shape)
        return (
            self.interp1d(
                self.y,
                self.x,
                (self.batch_shape and y.flatten(start_dim=nextradim) or y)
                .flatten(end_dim=nextradim - 1)
                .t(),
            )
            .t()
            .reshape(y.shape[:nextradim] + self.batch_shape)
        )

    def icdf(self, x: torch.Tensor):
        return self.ppf(x).reshape(x.shape)

    def ppf(self, x, right_sample_dims_count=0) -> torch.Tensor:
        """

        Parameters
        ----------
        x: torch.tensor
            the shape should either be (self.batch_shape..., ...) or
            (..., self.batch_shape..., sample dims...), where there are
            right_sample_dims_count sample dimensions to the right of
            self._prob_shape.
        right_sample_dims_count

        Returns
        -------
            The ppf at the given input, using the correctly broadcasted pdfs.
        """

        permute = not (
            self.batch_shape == x.shape[: len(self.batch_shape)]
            or (len(x.shape) == 2 and self.batch_shape.numel() == x.shape[0])
        )
        _x = (
            x.permute(
                *range(-len(self.batch_shape), -right_sample_dims_count),
                *range(len(x.shape) - right_sample_dims_count - len(self.batch_shape)),
                *range(-right_sample_dims_count, 0)
            )
            if permute
            else x
        )

        res = self.interp1d(
            self.x, self.y, _x.reshape((self.batch_shape.numel(), -1))
        ).reshape(_x.shape)
        return (
            res.permute(
                *range(len(self.batch_shape), len(x.shape) - right_sample_dims_count),
                *range(len(self.batch_shape)),
                *range(-right_sample_dims_count, 0)
            )
            if permute
            else res
        ).reshape(x.shape)

    def to_original(self, value: torch.Tensor):
        return self.sampling_distribution.icdf(self.cdf(value))

    def rsample(self, sample_shape=torch.Size()) -> TransformedSample:
        sample_shape = torch.Size(sample_shape)
        xnew = self.sampling_distribution.sample(
            (self.batch_shape.numel(), sample_shape.numel())
        )
        xtrans = self.sampling_distribution.cdf(xnew)
        out = self.ppf(xtrans).permute(1, 0).reshape(self.shape(sample_shape))
        return TransformedSample(out, xnew.permute(1, 0).reshape_as(out))

    def log_prob(self, value: TransformedSample):
        return (
            self.sampling_distribution.log_prob(
                value.orig_sample
                if isinstance(value, TransformedSample)
                else self.to_original(value)
            )
            if self.log_prob_of_original
            else self._log_prob(value) - self.log_scale
        )


class UNITSDistribution(InverseTransformDistribution):
    @staticmethod
    def make_sampling_distribution(dtype, device) -> torch.distributions.Distribution:
        return torch.distributions.Normal(
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
        )


