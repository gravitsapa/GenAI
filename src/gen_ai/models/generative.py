from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from gen_ai.metadata.configuration_collector import ContainingConfiguration


ImageShape = tuple[int, int]


class ImageGenerativeModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def sample(self, batch_size: int) -> Tensor:
        pass


class DescribedImageGenerativeModel(ContainingConfiguration, ImageGenerativeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
