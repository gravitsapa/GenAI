from typing import Optional
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.exceptions import require, ModelShapeError
from gen_ai.models.generative import ImageShape
from gen_ai.models.eff_vdvae.cnn import conv2d
from gen_ai.models.eff_vdvae.block_down import BlockDown, ResConvCellCommonInBlockDown


@dataclass(kw_only=True)
class TopDownBlocksConfig:
    n_residual_conv_cells: int
    res_conv_common: ResConvCellCommonInBlockDown


class TopDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_blocks_down: int,
        blocks_config: TopDownBlocksConfig,
        stride: int,
        skip_channels: int,
        latent_variates: int,
        out_channels: Optional[int]=None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.upsample_block = BlockDown(
            in_channels=in_channels,
            out_channels=out_channels,
            n_residual_conv_cells=blocks_config.n_residual_conv_cells,
            residual_conv_cell_config=blocks_config.res_conv_common,
            skip_channels=skip_channels,
            latent_variates=latent_variates,
            stride=stride,
        )

        self.blocks_down = nn.ModuleList([
            BlockDown(
                in_channels=out_channels,
                out_channels=out_channels,
                n_residual_conv_cells=blocks_config.n_residual_conv_cells,
                residual_conv_cell_config=blocks_config.res_conv_common,
                skip_channels=skip_channels,
                latent_variates=latent_variates,
                stride=1,
            )
            for _ in range(n_blocks_down)
        ])

    def forward(
        self,
        y: Tensor,
        x_skip: Tensor,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]:
        posterior_params_list = []
        prior_params_list = []

        y, posterior_params, prior_params = self.upsample_block(y, x_skip)

        posterior_params_list.append(posterior_params)
        prior_params_list.append(prior_params)

        for block_down in self.blocks_down:
            y, posterior_params, prior_params = block_down(y, x_skip)
                
            posterior_params_list.append(posterior_params)
            prior_params_list.append(prior_params)

        return y, posterior_params_list, prior_params_list

    def sample_from_prior(
        self,
        y: Tensor,
        temperature: float,
    ) -> tuple[Tensor, list[Tensor]]:
        prior_zs = []

        y, prior_z = self.upsample_block.sample_from_prior(y, temperature=temperature)

        prior_zs.append(prior_z)

        for block_down in self.blocks_down:
            y, prior_z = block_down.sample_from_prior(y, temperature=temperature)

            prior_zs.append(prior_z)

        return y, prior_zs
        


@dataclass(kw_only=True)
class TopDownBlocksCommon:
    n_blocks_down: int
    blocks_config: TopDownBlocksConfig


class TopDown(nn.Module):
    def __init__(
        self,
        image_shape: ImageShape,
        blocks_common_config: TopDownBlocksCommon,
        blocks_channels: tuple[int, ...],
        blocks_stride: tuple[int, ...],
        blocks_skip_channels: tuple[int, ...],
        blocks_latent_variates: tuple[int, ...],
        out_channels: int = 3,
        out_conv_kernel: int = 3,
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

        stride_prod = np.prod(blocks_stride)

        require(
            image_shape[0] % stride_prod == 0 and image_shape[1] % stride_prod == 0,
            ModelShapeError,
            lambda: (
                "image_shape dimensions must be divisible by the product of "
                f"blocks_stride ({stride_prod}), got {image_shape}"
            ),
        )

        top_latent_shape = (1, blocks_channels[0]) + \
            tuple(np.array(image_shape, dtype=int) // stride_prod)
        self.trainable_h = torch.nn.Parameter(
            data=torch.empty(top_latent_shape),
            requires_grad=True
        )
        nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        
        self.blocks = nn.ModuleList()

        blocks_in_channels = blocks_channels[0:1] + blocks_channels[:-1]

        for (
            block_in_channels, 
            block_out_channels, 
            block_stride,
            block_skip_channels,
            block_latent_variates,
        ) in zip(
            blocks_in_channels,
            blocks_channels,
            blocks_stride,
            blocks_skip_channels,
            blocks_latent_variates,
            strict=True,
        ):
            self.blocks.append(
                TopDownBlock(
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    n_blocks_down=blocks_common_config.n_blocks_down,
                    blocks_config=blocks_common_config.blocks_config,
                    stride=block_stride,
                    skip_channels=block_skip_channels,
                    latent_variates=block_latent_variates,
                )
            )

        self.output_conv = conv2d(
            in_channels=blocks_channels[-1],
            out_channels=out_channels,
            kernel_size=out_conv_kernel,
        )


    def forward(
        self, 
        skip_list: list[Tensor],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]:
        batch_size = skip_list[0].size()[0]
        y = torch.tile(self.trainable_h, (batch_size, 1, 1, 1))

        posterior_params_list = []
        prior_params_list = []

        for block, x_skip in zip(
            self.blocks,
            skip_list,
            strict=True,
        ):
            y, new_posterior_params_list, new_prior_params_list = block(y, x_skip)

            posterior_params_list.extend(new_posterior_params_list)
            prior_params_list.extend(new_prior_params_list)

        y = self.output_conv(y)

        return y, posterior_params_list, prior_params_list

    @torch.inference_mode()
    def sample_from_prior(
        self, 
        batch_size: int, 
        temperature: float,
    ) -> tuple[Tensor, list[Tensor]]:
        y = torch.tile(self.trainable_h, (batch_size, 1, 1, 1))

        prior_zs = []
        for block in self.blocks:
            y, new_prior_zs = block.sample_from_prior(y, temperature=temperature)

            prior_zs.extend(new_prior_zs)

        y = self.output_conv(y)

        return y, prior_zs

        
