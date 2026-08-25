import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.exceptions import require, ModelShapeError
from gen_ai.models.generative import ImageShape
from gen_ai.models.eff_vdvae.top_down import (
    TopDown,
    TopDownBlocksCommon,
    TopDownBlocksConfig,
    ResConvCellCommonInBlockDown,
)

from gen_ai.models.eff_vdvae.bottom_up import (
    BottomUp, 
    BottomUpBlocksCommon, 
    BottomUpBlocksConfig,
    ResConvCellCommonInBlocksUp
)


class EffVDVAE(nn.Module):
    def __init__(
        self,
        image_shape: ImageShape,
        n_layers_in_block: int,
        blocks_channels_bottom_up: tuple[int, ...],
        blocks_strides_bottom_up: tuple[int, ...],
        blocks_skip_channels: tuple[int, ...],
        blocks_latent_variates: tuple[int, ...],
        n_output_mixtures: int,
        in_channels: int = 3,
        in_kernel: int = 1,
        out_kernel: int = 1,
        n_residual_conv_cells_in_layer: int = 1,
        n_conv_layers_in_residual: int = 2,
        kernel_size: int = 3,
        bottleneck_channels_ratio: float = 0.25,
    ):
        super().__init__()

        blocks_cnt = len(blocks_channels_bottom_up)
        require(
            blocks_cnt == len(blocks_strides_bottom_up),
            ModelShapeError,
            "blocks_channels_bottom_up and blocks_strides_bottom_up must have the same length, "
            f"got {blocks_cnt} and {len(blocks_strides_bottom_up)}",
        )

        init_scaler = np.sqrt(1. / ((1 + n_layers_in_block) * blocks_cnt))

        bottom_up_common_config = BottomUpBlocksCommon(
            n_blocks_up=n_layers_in_block,
            blocks_config=BottomUpBlocksConfig(
                n_residual_conv_cells=n_residual_conv_cells_in_layer,
                res_conv_common=ResConvCellCommonInBlocksUp(
                    kernel_size=kernel_size,
                    n_layers=n_conv_layers_in_residual,
                    init_scaler=init_scaler,
                    bottleneck_channels_ratio=bottleneck_channels_ratio,
                )
            )
        )

        self.bottom_up = BottomUp(
            blocks_common_config=bottom_up_common_config,
            blocks_channels=blocks_channels_bottom_up,
            blocks_stride=blocks_strides_bottom_up,
            blocks_skip_channels=blocks_skip_channels,
            in_channels=in_channels,
            in_conv_kernel=in_kernel,
        )

        top_down_common_config = TopDownBlocksCommon(
            n_blocks_down=n_layers_in_block,
            blocks_config=TopDownBlocksConfig(
                n_residual_conv_cells=n_residual_conv_cells_in_layer,
                res_conv_common=ResConvCellCommonInBlockDown(
                    kernel_size=kernel_size,
                    n_layers=n_conv_layers_in_residual,
                    init_scaler=init_scaler,
                    bottleneck_channels_ratio=bottleneck_channels_ratio,
                ),
            )
        )

        out_channels = n_output_mixtures * (in_channels * 3 + 1)

        self.top_down = TopDown(
            image_shape=image_shape,
            blocks_common_config=top_down_common_config,
            blocks_channels=blocks_channels_bottom_up[::-1],
            blocks_stride=blocks_strides_bottom_up[::-1],
            blocks_skip_channels=blocks_skip_channels[::-1],
            blocks_latent_variates=blocks_latent_variates[::-1],
            out_channels=out_channels,
            out_conv_kernel=out_kernel,
        )

    def forward(self, intput_tensor: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]:
        skip_list = self.bottom_up(intput_tensor)
        output_tensor, posterior_params_list, prior_params_list = self.top_down(skip_list[::-1])

        return output_tensor, posterior_params_list, prior_params_list
