import typing
from math import pi
from warnings import warn

import torch

from .torchutils import broadcast_gather, pad_dims


try:
    from pykeops.torch import LazyTensor
except ImportError:
    warn('KeOps not available.', category=ImportWarning)


def d2_kernel(x: torch.Tensor, y: torch.Tensor) -> LazyTensor:
    """
    Return a keops.LazyTensor of squared distances between x and y.

    Parameters
    ----------
    x: torch.Tensor: Size(batch dims..., N, ndim)
    y: torch.Tensor: Size(batch dims..., M, ndim)

    Returns
    -------
        torch.Tensor: Size(batch_dims..., N, M)
    Return d2 such that d2[..., i, j] = sqdist(x[..., i], y[..., j]).
    """
    return LazyTensor.sqdist(
        LazyTensor(x.unsqueeze(-2)),
        LazyTensor(y.unsqueeze(-3))
    )


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma2: torch.Tensor,
               return_s: bool = False) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
    """
    Return a Gaussian kernel between x and y with a (possibly batched)
    lengthscale sigma, normalized to unit variance.

    Parameters
    ----------
    x: torch.Tensor: Size(batch_dims..., N, ndim)
    y: torch.tesnor: Size(batch_dims..., M, ndim)
    sigma2: torch.Tensor
        Shape must be alignable to the left with the dimensions of x:
        sigma.shape == x.shape[:sigma.ndim] or broadcastable
    return_s: bool
        whether to return properly broadcasted 0.5 / sigma**2

    Returns
    -------
        torch.Tensor: Size(batch_dims..., N, M)
    """
    # Align sigma to leftmost dims of x, and add a dim for y
    s = (0.5 / sigma2).reshape(*sigma2.shape, *((1,) * (x.ndim + 1 - sigma2.ndim)))
    ret = (- d2_kernel(x, y) * s).exp()
    return ret if not return_s else (ret, s)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma2: torch.Tensor) -> LazyTensor:
    """
    Return a Gaussian kernel between x and y with a (possibly batched)
    lengthscale sigma, normalized such that integrating over x would give one.

    Parameters
    ----------
    x: torch.Tensor: Size(batch_dims..., N, ndim)
    y: torch.tesnor: Size(batch_dims..., M, ndim)
    sigma2: torch.Tensor
        Shape must be alignable to the left with the dimensions of x:
        sigma.shape == x.shape[:sigma.ndim] or broadcastable

    Returns
    -------
        torch.Tensor: Size(batch_dims..., N, M)
    """
    ret, s = rbf_kernel(x, y, sigma2, True)
    return ret * (s/pi)**(x.shape[-1]/2)


def kNN(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """
    Get k nearest neighbours using keops.

    Parameters
    ----------
    x: torch.Tensor: Size(batch dims..., N, ndim)
    y: torch.Tensor: Size(batch dims..., M, ndim)
    k: int

    Returns
    -------
        torch.Tensor(dtype=int): Size(batch_dims..., N, K)
    Returns idx such that y[a, b..., idx[a, b..., i]] is
    an array of the K points in y[a, b...] nearest to x[a, b..., i], where
    a, b... are indices into the batch dimensions. Note that if x == y, then
    idx[..., i][0] == i, i.e. if the two sets of points are the same, the
    nearest neighbour to each point is the point itself.
    """
    x, y = pad_dims(x, y)
    return LazyTensor.sqdist(
        LazyTensor(x.unsqueeze(-2)),
        LazyTensor(y.unsqueeze(-3))
    ).argKmin(k, dim=len(x.shape) - 1)  # reduce along x dimension


def kNN_d2(x: torch.Tensor, y: torch.Tensor, k: int,
           x0: torch.Tensor = None, y0: torch.Tensor = None) -> torch.Tensor:
    """
    Get squared distances to k nearest neighbours using keops.

    Parameters
    ----------
    x: torch.Tensor: Size(batch dims..., M, ndim)
    y: torch.Tensor: Size(batch dims..., N, ndim)
    k: int
    x0: torch.Tensor: Size(batch dims..., M, ndim)
    y0: torch.Tensor: Size(batch dims..., N, ndim)

    Returns
    -------
    Returns d2 such that d2[a, b..., i, j] is the distance squared between
    x[a, b..., i] and its j-th nearest neighbour among y[a, b...], where
    a, b... are indices into the batch dimensions. Note that if x == y, then
    d2[..., 0] == 0, i.e. if the two point sets are the same, the nearest
    neighbour to each point is the point itself.
    If x0, y0 are given, they are used to determine the neighbours.
    """
    if x0 is None or y0 is None:
        x0 = x
        y0 = y

    return (x.unsqueeze(-2) - broadcast_gather(y, -2, kNN(x0, y0, k), index_ndim=2)).pow(2.).sum(-1)
