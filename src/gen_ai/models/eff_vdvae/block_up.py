from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.models.common import ModuleFactory
from gen_ai.models.eff_vdvae.cnn import conv2d, ResidualConvCell


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        non_linearity: ModuleFactory=lambda: nn.LeakyReLU(negative_slope=0.1),
    ):
        super().__init__()

        self.conv = conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride,
            stride=stride,
        )
        self.nonlin = non_linearity()

    def forward(self, input_tensor):
        return self.nonlin(self.conv(input_tensor))


@dataclass(kw_only=True)
class ResConvCellCommonInBlocksUp:
    kernel_size: int | tuple[int, int]
    n_layers: int
    init_scaler: float
    bottleneck_channels_ratio: float = 0.25


class BlockUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_residual_conv_cells: int,
        residual_conv_cell_config: ResConvCellCommonInBlocksUp,
        out_channels: Optional[int]=None,
        stride: int = 1,
        compute_skip: bool = False,
        skip_channels: Optional[int] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        
        self.residual_block = nn.Sequential(*[
            ResidualConvCell(
                kernel_size=residual_conv_cell_config.kernel_size,
                n_layers=residual_conv_cell_config.n_layers,
                bottleneck_channels_ratio=residual_conv_cell_config.bottleneck_channels_ratio,
                init_scaler=residual_conv_cell_config.init_scaler,
                in_channels=in_channels,
                out_channels=in_channels,
            )
            for _ in range(n_residual_conv_cells)
        ])
        
        self.compute_skip = compute_skip
        if self.compute_skip:
            if skip_channels is None:
                skip_channels = in_channels

            self.skip_projection = conv2d(
                in_channels=in_channels,
                out_channels=skip_channels,
                kernel_size=1,
            )

        if stride > 1:
            self.resample = Downsample(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        elif in_channels != out_channels:
            self.resample = ResidualConvCell(
                kernel_size=1,
                n_layers=residual_conv_cell_config.n_layers,
                bottleneck_channels_ratio=residual_conv_cell_config.bottleneck_channels_ratio,
                init_scaler=residual_conv_cell_config.init_scaler,
                in_channels=in_channels,
                out_channels=out_channels,
            )
        else:
            self.resample = nn.Identity()

    def forward(self, input_tensor) -> Tensor | tuple[Tensor, Tensor]:
        pre_skip = self.residual_block(input_tensor)
        output_tensor = self.resample(pre_skip)

        if self.compute_skip:
            skip_output = self.skip_projection(pre_skip)
            return output_tensor, skip_output

        return output_tensor
