from typing import Any, Callable, Optional, Union
import torch.nn as nn
from torch import device

ModuleFactory = Callable[..., nn.Module]

def get_model_device(model: nn.Module) -> device:
    return next(model.parameters()).device
