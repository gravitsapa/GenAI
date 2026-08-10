from typing import Any, Callable, Optional, Union
import torch.nn as nn

ModuleFactory = Callable[..., nn.Module]