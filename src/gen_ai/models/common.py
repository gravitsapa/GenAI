from typing import Any, Callable, Optional, Union
import torch.nn as nn

ModuleFactory = Callable[..., nn.Module]

def get_model_device(model: nn.Module):
    return next(model.parameters()).device
