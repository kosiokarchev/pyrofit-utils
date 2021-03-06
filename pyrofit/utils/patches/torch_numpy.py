"""
torch.Tensors are automatically detached and copied to cpu (if needed)
when NumPy access is requested.
"""
import torch

if not hasattr(torch.Tensor, '_numpy'):
    torch.Tensor._numpy = torch.Tensor.numpy
    torch.Tensor.numpy = lambda self: self.detach().cpu()._numpy()
