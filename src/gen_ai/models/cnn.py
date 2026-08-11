from typing import Any, Callable, Optional, Union
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from gen_ai.models.common import ModuleFactory
from gen_ai.models.upsampling import Upsample2D
from gen_ai.models.downsampling import Downsample2D
from gen_ai.models.normalization import NormalizationFactory, GroupNormalizationFactory


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int=1,
    groups: int=1,
    dilation: int=1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int=1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResChange(Enum):
    UP = 1
    IDENTIAL = 2
    DOWN = 3


class ResNetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_change: ResChange,
        dropout: float=0.0,
        non_linearity: ModuleFactory=nn.SiLU,
        upsampler: ModuleFactory=Upsample2D,
        downsampler: ModuleFactory=Downsample2D,
        normalization: NormalizationFactory=GroupNormalizationFactory(8),
        normalization_out: Optional[NormalizationFactory]=None,
    ):
        super().__init__()

        if normalization_out is None:
            normalization_out = normalization

        self.res_change = res_change

        self.norm1 = normalization(in_channels)
        self.nonlinearity1 = non_linearity()

        if self.res_change == ResChange.UP:
            self.resample_hidden = upsampler(in_channels)
            self.resample_input = upsampler(in_channels)
        elif self.res_change == ResChange.DOWN:
            self.resample_hidden = downsampler(in_channels)
            self.resample_input = downsampler(in_channels)
        else:
            self.resample_hidden = nn.Identity()
            self.resample_input = nn.Identity()

        self.conv_shortcut = conv1x1(
            in_channels,
            out_channels
        )

        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm2 = normalization_out(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3(out_channels, out_channels)

        self.nonlinearity2 = non_linearity()


    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = input_tensor
        skip_connection = input_tensor

        hidden = self.norm1(hidden)
        hidden = self.nonlinearity1(hidden)
        hidden = self.resample_hidden(hidden)
        skip_connection = self.resample_input(skip_connection)
        hidden = self.conv1(hidden)

        hidden = self.norm2(hidden)
        hidden = self.nonlinearity2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)

        skip_connection = self.conv_shortcut(skip_connection)

        output_tensor = skip_connection + hidden

        return output_tensor

# Sources:
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py