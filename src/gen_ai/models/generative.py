from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from gen_ai.metadata.collectors import DeclarationDescribed, RuntimeDescribed


class ImageGenerativeModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def sample(self, batch_size: int) -> Tensor:
        pass


class DescribedImageGenerativeModel(ImageGenerativeModel, DeclarationDescribed):
    pass
