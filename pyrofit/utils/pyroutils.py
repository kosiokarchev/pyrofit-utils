import typing as tp
from operator import setitem

import pyro
import torch

__all__ = 'apply_across_trace',


def apply_across_trace(trace: pyro.poutine.Trace, func: tp.Callable[[torch.Tensor], torch.Tensor]):
    for node in trace.nodes.values():
        for key in ('value', 'log_prob'):
            key in node and setitem(node, key, func(node[key]))
