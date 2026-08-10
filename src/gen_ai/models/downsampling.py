import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Optional


class Downsample2D(nn.Module):
    def __init__(
        self, 
        channels: int,
        out_channels: Optional[int]=None,
        kernel_size: int=3,
        stride: int=2,
        padding: int=1,
        bias: bool=True
    ):
        self.channels = channels
        self.out_channels = channels if out_channels is None else out_channels

        self.conv = nn.Conv2d(
            self.channels, 
            self.out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        assert input_tensor.shape[1] == self.channels
        return self.conv(input_tensor)
