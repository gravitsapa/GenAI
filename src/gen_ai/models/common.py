from typing import Any, Callable, Optional, Union
import torch.nn as nn
from torch import Device

ModuleFactory = Callable[..., nn.Module]

def get_model_device(model: nn.Module) -> Device:
    return next(model.parameters()).device
