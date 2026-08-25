from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

ModuleFactory = Callable[..., nn.Module]

def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def one_hot(indices: Tensor, depth: int, dim: int, device: torch.device):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=device)
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot