from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.exceptions import require, ModelShapeError
from gen_ai.models.eff_vdvae.cnn import Conv2dWithZeroBias
from gen_ai.models.eff_vdvae.block_up import BlockUp, ResConvCellCommonInBlocksUp


@dataclass(kw_only=True)
class BottomUpBlocksConfig:
    n_residual_conv_cells: int
    res_conv_common: ResConvCellCommonInBlocksUp


class BottomUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_blocks_up: int,
        blocks_config: BottomUpBlocksConfig,
        stride: int,
        skip_channels: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.blocks_up = nn.Sequential(*[
            BlockUp(
                in_channels=in_channels,
                n_residual_conv_cells=blocks_config.n_residual_conv_cells,
                residual_conv_cell_config=blocks_config.res_conv_common,
                compute_skip=False,
                stride=1,
            )
            for _ in range(n_blocks_up)
        ])

        self.downsample_block = BlockUp(
            in_channels=in_channels,
            out_channels=out_channels,
            n_residual_conv_cells=blocks_config.n_residual_conv_cells,
            residual_conv_cell_config=blocks_config.res_conv_common,
            compute_skip=True,
            skip_channels=skip_channels,
            stride=stride,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.blocks_up(x)
        output_tensor, skip_output = self.downsample_block(x)

        return output_tensor, skip_output


@dataclass(kw_only=True)
class BottomUpBlocksCommon:
    blocks_config: BottomUpBlocksConfig


class BottomUp(nn.Module):
    def __init__(
        self,
        blocks_common_config: BottomUpBlocksCommon,
        n_layers_in_block: tuple[int, ...],
        blocks_channels: tuple[int, ...],
        blocks_stride: tuple[int, ...],
        blocks_skip_channels: tuple[int, ...],
        in_channels: int = 3,
        in_conv_kernel: int = 3,
    ):
        super().__init__()

        require(
            len(blocks_channels) == len(blocks_stride),
            ModelShapeError,
            lambda: (
                "blocks_channels and blocks_stride must have the same length, got "
                f"{len(blocks_channels)} and {len(blocks_stride)}"
            ),
        )

        self.input_conv = Conv2dWithZeroBias(
            in_channels=in_channels,
            out_channels=blocks_channels[0],
            kernel_size=in_conv_kernel,
        )

        self.blocks = nn.ModuleList()

        blocks_out_channels = blocks_channels[1:] + blocks_channels[-1:]

        for (
            n_layers,
            block_in_channels, 
            block_out_channels, 
            block_stride, 
            block_skip_channels
        ) in zip(
            n_layers_in_block,
            blocks_channels,
            blocks_out_channels,
            blocks_stride,
            blocks_skip_channels,
            strict=True,
        ):
            self.blocks.append(
                BottomUpBlock(
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    n_blocks_up=n_layers,
                    blocks_config=blocks_common_config.blocks_config,
                    stride=block_stride,
                    skip_channels=block_skip_channels,
                )
            )

    def forward(self, input_tensor: Tensor) -> list[Tensor]:
        x = input_tensor
        x = self.input_conv(x)

        skip_list = []

        for block in self.blocks:
            x, skip_output = block(x)
            skip_list.append(skip_output)

        return skip_list

