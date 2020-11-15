from math import gamma, log, pi

import torch

from .keopsutils import broadcast_index, kNN, kNN_d2


class Entropy:
    def __init__(self, k=50, kmin=4, scale=20, power=0.33, device='cpu'):
        self.kmin, self.k = kmin, k
        self.weights = self.get_weights(k, kmin, scale, power, device=device)
        self.device = device

    @staticmethod
    def get_weights(k=50, kmin=4, scale=20, power=1., device='cpu'):
        w = torch.arange(float(kmin), float(k)+1)
        w = w**power * torch.exp(-w / scale)
        weights = torch.zeros(k, device=device)
        weights[kmin-1:] = w / w.sum()
        return weights

    @staticmethod
    def ballfactor(ndim):
        return pi**(ndim/2) / gamma(1 + ndim/2)

    def normalising_factor(self, N, ndim):
        return ((torch.digamma(torch.arange(1., self.k+1, device=self.device)) * self.weights).sum()
                - torch.digamma(torch.tensor(float(N)))
                - log(self.ballfactor(ndim)))

    def d2(self, x):
        return kNN_d2(x, x, len(self.weights) + 1)[..., :, 1:]  # [0] was the point itself

    def log_p(self, x, d_min=1e-3):
        ndim = x.shape[-1]
        return ndim / 2 * (self.weights * torch.log(self.d2(x) + d_min**2)).sum(-1)

    def entropy_loss(self, x, d_min=1e-3):
        return - self.log_p(x, d_min).sum(-1)

    def full_entropy(self, x, d_min=1e-3):
        N, ndim = x.shape[-2:]
        return self.log_p(x, d_min).mean(-1) - self.normalising_factor(N, ndim)


class ConstantNeighboursEntropy(Entropy):
    def __init__(self, *args, x0, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = kNN(x0, x0, len(self.weights)+1)

    def d2(self, x):
        return (x.unsqueeze(-2) - broadcast_index(x, self.i)).pow(2.).sum(-1)[..., :, 1:]
