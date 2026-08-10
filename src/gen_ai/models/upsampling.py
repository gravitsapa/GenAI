import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class Upsample2D(nn.Module):
    def __init__(
        self, 
        channels: int,
    ):
        self.channels = channels

    def forward(self, input_tensor: Tensor) -> Tensor:
        assert input_tensor.shape[1] == self.channels
        return F.interpolate(input_tensor, scale_factor=2, mode="nearest")
