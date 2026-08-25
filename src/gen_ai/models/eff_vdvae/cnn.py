from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.models.common import ModuleFactory


def conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int=1,
    padding: int | str = 'same',
    bias: bool=True,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        padding=padding,
    )


class ResidualConvCell(nn.Module):
    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        bottleneck_channels_ratio: float = 0.25,
        out_channels: Optional[int] = None,
        init_scaler: float = 1.0,
        kernel_size: int | tuple[int, int] = 3,
        use_1x1_cells: bool = True,
        non_linearity: ModuleFactory=nn.SiLU,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        bottleneck_channels = int(in_channels * bottleneck_channels_ratio)

        self.convs = nn.Sequential()
        self.convs.extend([
            non_linearity(),
            conv2d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1 if use_1x1_cells else 3,
            )
        ])

        for _ in range(n_layers):
            self.convs.extend([
                non_linearity(),
                conv2d(
                    in_channels=bottleneck_channels,
                    out_channels=bottleneck_channels,
                    kernel_size=kernel_size,
                )
            ])

        last_conv = conv2d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1 if use_1x1_cells else 3,
        )
        last_conv.weight.data *= init_scaler

        self.convs.extend([
            non_linearity(),
            last_conv
        ])

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )

    def forward(
        self,
        input_tensor
    ):
        return self.convs(input_tensor) + self.residual(input_tensor)

