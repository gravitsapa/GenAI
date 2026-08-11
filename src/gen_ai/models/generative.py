from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class ImageGenerativeModel(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def sample(self, batch_size: int) -> Tensor:
        pass
