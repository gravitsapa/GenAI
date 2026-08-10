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

        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm2 = normalization_out(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3(out_channels, out_channels)

        self.nonlinearity2 = non_linearity()


    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = input_tensor

        hidden = self.norm1(hidden)
        hidden = self.nonlinearity1(hidden)
        hidden = self.resample_hidden(hidden)
        input_tensor = self.resample_input(input_tensor)
        hidden = self.conv1(hidden)

        hidden = self.norm2(hidden)
        hidden = self.nonlinearity2()
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)

        output_tensor = input_tensor + hidden

        return output_tensor


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        out_channels: int=3,
        block_channels: tuple[int, ...]=(64,),
        norm_num_groups: int=8,
        activation_fn: ModuleFactory=nn.SiLU,
        mid_layers: int=2,
        double_output: bool=False,
    ):
        super().__init__()

        self.conv_in = conv3x3(
            in_channels,
            block_channels[0]
        )

        self.down_blocks = nn.Sequential()
        last_channels = block_channels[0]
        
        for block_num, channels in enumerate(block_channels):
            block_in_channels = last_channels
            block_out_channels = channels

            last_channels = block_out_channels
            is_last_block = block_num == len(block_channels) - 1

            down_block = ResNetBlock2D(
                block_in_channels,
                block_out_channels,
                ResChange.IDENTIAL if is_last_block else ResChange.DOWN,
                non_linearity=activation_fn,
                normalization=GroupNormalizationFactory(norm_num_groups),
            )

            self.down_blocks.append(down_block)

        self.mid_blocks = nn.Sequential()
        for block_num in range(mid_layers):
            self.mid_blocks.append(ResNetBlock2D(
                in_channels=last_channels,
                out_channels=last_channels,
                res_change=ResChange.IDENTIAL,
                non_linearity=activation_fn,
                normalization=GroupNormalizationFactory(norm_num_groups),
            ))

        self.conv_norm_out = GroupNormalizationFactory(norm_num_groups)(last_channels)
        self.conv_act = activation_fn()
        
        conv_out_channels = 2 * out_channels if double_output else out_channels
        self.conv_out = conv3x3(last_channels, conv_out_channels)


    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = input_tensor

        hidden = self.conv_in(hidden)
        hidden = self.down_blocks(hidden)

        hidden = self.mid_blocks(hidden)

        hidden = self.conv_norm_out(hidden)
        hidden = self.conv_act(hidden)
        output_tensor = self.conv_out(hidden)

        return output_tensor


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        out_channels: int=3,
        block_channels: tuple[int, ...]=(64,),
        norm_num_groups: int=8,
        activation_fn: ModuleFactory=nn.SiLU,
        mid_layers: int=2,
    ):
        super().__init__()

        self.conv_in = conv3x3(
            in_channels,
            block_channels[-1]
        )

        last_channels = block_channels[-1]

        self.mid_blocks = nn.Sequential()
        for block_num in range(mid_layers):
            self.mid_blocks.append(ResNetBlock2D(
                in_channels=last_channels,
                out_channels=last_channels,
                res_change=ResChange.IDENTIAL,
                non_linearity=activation_fn,
                normalization=GroupNormalizationFactory(norm_num_groups),
            ))

        self.up_blocks = nn.Sequential()

        for block_num, channels in enumerate(reversed(block_channels)):
            block_in_channels = last_channels
            block_out_channels = channels

            last_channels = block_out_channels
            is_last_block = block_num == len(block_channels) - 1

            up_block = ResNetBlock2D(
                block_in_channels,
                block_out_channels,
                ResChange.IDENTIAL if is_last_block else ResChange.UP,
                non_linearity=activation_fn,
                normalization=GroupNormalizationFactory(norm_num_groups),
            )

            self.up_blocks.append(up_block)

        self.conv_norm_out = GroupNormalizationFactory(norm_num_groups)(last_channels)
        self.conv_act = activation_fn()

        self.conv_out = conv3x3(last_channels, out_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = input_tensor

        hidden = self.conv_in(hidden)
        hidden = self.mid_blocks(hidden)
        hidden = self.up_blocks(hidden)

        hidden = self.conv_norm_out(hidden)
        hidden = self.conv_act(hidden)
        ouput_tensor = self.conv_out(hidden)

        return ouput_tensor
        

# Sources:
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py