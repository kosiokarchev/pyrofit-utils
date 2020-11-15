import torch
from pyro.distributions import TorchDistribution, constraints

from .. import _size, _Distribution


class DistributionWrapper(TorchDistribution):
    """Base class for a distribution that delegates to another one."""

    arg_constraints = {}

    def __init__(
        self,
        base_dist: _Distribution,
        batch_shape: _size = None,
        event_shape: _size = None,
        validate_args=None,
    ):
        super().__init__(
            batch_shape and batch_shape or base_dist.batch_shape,
            event_shape and event_shape or base_dist.event_shape,
            validate_args=validate_args,
        )
        self.base_dist = base_dist

    @property
    def batch_dim(self):
        return len(self.batch_shape)

    def expand(self, batch_shape, _instance=None):
        new = self.__new__(type(self)) if _instance is None else _instance
        new.base_dist = self.base_dist.expand(torch.Size(batch_shape))
        super(DistributionWrapper, new).__init__(batch_shape, self.event_shape, False)
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    def entropy(self):
        return self.base_dist.entropy()

    def enumerate_support(self, expand=True):
        return self.base_dist.enumerate_support(expand=expand)


class ExtraIndependent(DistributionWrapper):
    """
    Add more event dimensions without consuming existing batch dimensions.

    Useful in order to treat multiple samples as a dimension in an event and
    sum them in the probability and to allow proper plating.
    """

    def __init__(
        self, base_dist: _Distribution, event_shape: _size, validate_args=None
    ):
        self.sample_shape = torch.Size(event_shape)

        super().__init__(
            base_dist,
            validate_args=validate_args,
            event_shape=self.sample_shape + base_dist.event_shape,
        )

    @property
    def sample_dim(self):
        return len(self.sample_shape)

    def expand(self, batch_shape, _instance=None):
        new = super().expand(batch_shape, _instance)
        new.sample_shape = self.sample_shape
        return new

    def roll_to_right(self, value: torch.Tensor):
        return value.permute(
            tuple(range(self.sample_dim, self.sample_dim + self.batch_dim))
            + tuple(range(self.sample_dim))
            + tuple(range(-self.event_dim + self.sample_dim, 0))
        )

    def roll_to_left(self, value: torch.Tensor):
        return value.permute(
            tuple(range(self.batch_dim, self.batch_dim + self.sample_dim))
            + tuple(range(self.batch_dim))
            + tuple(range(-self.event_dim + self.sample_dim, 0))
        )

    def sample(self, sample_shape: _size = torch.Size()):
        return self.roll_to_right(
            self.base_dist.sample(self.sample_shape + sample_shape)
        )

    def rsample(self, sample_shape: _size = torch.Size()):
        return self.roll_to_right(
            self.base_dist.rsample(self.sample_shape + sample_shape)
        )

    def log_prob(self, value):
        return self.base_dist.log_prob(self.roll_to_left(value)).sum(
            tuple(range(self.sample_dim))
        )

    def entropy(self):
        return self.base_dist.entropy()

