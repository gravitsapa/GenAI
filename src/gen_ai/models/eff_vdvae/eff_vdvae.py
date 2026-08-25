import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.exceptions import require, ModelShapeError
from gen_ai.models.common import get_model_device, one_hot
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
        input_min_value: float = -1.,
        input_max_value: float = 1.,
        in_channels: int = 3,
        in_kernel: int = 1,
        out_kernel: int = 1,
        n_residual_conv_cells_in_layer: int = 1,
        n_conv_layers_in_residual: int = 2,
        kernel_size: int = 3,
        bottleneck_channels_ratio: float = 0.25,
    ):
        super().__init__()

        require(
            in_channels == 3,
            NotImplementedError,
            "Now implemented only if in_channels == 3."
        )

        require(
            np.isclose(input_min_value, -1.).item() and np.isclose(input_max_value, 1.).item(),
            NotImplementedError,
            "Now implemented onlu if image values in [-1, 1]"
        )

        self.input_min_value = input_min_value
        self.input_max_value = input_max_value

        blocks_cnt = len(blocks_channels_bottom_up)
        require(
            blocks_cnt == len(blocks_strides_bottom_up),
            ModelShapeError,
            "blocks_channels_bottom_up and blocks_strides_bottom_up must have the same length, "
            f"got {blocks_cnt} and {len(blocks_strides_bottom_up)}",
        )

        self.image_shape = image_shape

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

        self.n_output_mixtures = n_output_mixtures
        self.in_channels = in_channels
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

    
    @torch.inference_mode()
    def _sample_from_logits(
        self,
        logits: Tensor,
        smoothing_beta: float = np.log(2),
        min_scale: float = np.exp(-250),
        temperature: float = 1.,
    ) -> Tensor:
        device = get_model_device(self)
        
        n_mixtures = self.n_output_mixtures
        in_channels = self.in_channels

        B, X, H, W = logits.size()  # B, M*(3*C+1), H, W,
        require(
            X == n_mixtures * (3 * in_channels + 1),
            ValueError,
            lambda: (
                "Invalid logits channel count: expected "
                f"{n_mixtures * (3 * in_channels + 1)} for {n_mixtures} mixtures "
                f"and {in_channels} input channels, got {X}"
            ),
        )

        require(
            (H, W) == self.image_shape,
            ValueError,
            lambda: (
                f"Invalid logits spatial shape: expected {self.image_shape}, got {(H, W)}"
            ),
        )

        logit_probs = logits[:, :n_mixtures, :, :]  # B, M, H, W
        l = logits[:, n_mixtures:, :, :]  # B, M*C*3 ,H, W
        l = l.reshape(B, in_channels, 3 * n_mixtures, H, W)  # B, C, 3 * M, H, W

        model_means = l[:, :, :n_mixtures, :, :]  # B, C, M, H, W

        scales = l[:, :, n_mixtures: 2 * n_mixtures, :, :] # B, C, M, H, W
        softplus = nn.Softplus(beta=smoothing_beta)
        scales = torch.maximum(
            softplus(scales), 
            torch.as_tensor(min_scale, device=device),
        )

        model_coeffs = torch.tanh(
            l[:, :, 2 * n_mixtures: 3 * n_mixtures, :, :]
        )  # B, C, M, H, W

        uniform_min = 1e-5
        uniform_max = 1. - 1e-5
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(logit_probs, device=device) * (uniform_max - uniform_min) + uniform_min
        ))

        gumbel_trick_logits = logit_probs / temperature + gumbel_noise
        indices = torch.argmax(gumbel_trick_logits, dim=1)
        mask = one_hot(
            indices,
            depth=n_mixtures,
            dim=1,
            device=device,
        ).unsqueeze(1)   # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(model_means * mask, dim=2)  # B, C, H, W
        scales = torch.sum(scales * mask, dim=2)  # B, C, H, W
        coeffs = torch.sum(model_coeffs * mask, dim=2)  # B, C, H, W

        u = torch.rand_like(means, device=device) * (uniform_max - uniform_min) + uniform_min
        x = means + scales * temperature * (
                torch.log(u) - torch.log(1. - u))  # B, C, H, W

        # Autoregressively predict RGB
        x0 = torch.clamp(
            x[:, 0:1, :, :], 
            min=self.input_min_value, 
            max=self.input_max_value
        )  # B, 1, H, W
        x1 = torch.clamp(
            x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0, 
            min=self.input_min_value,
            max=self.input_max_value
        )  # B, 1, H, W
        x2 = torch.clamp(
            x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1,
            min=self.input_min_value,
            max=self.input_max_value
        )  # B, 1, H, W

        x = torch.cat([x0, x1, x2], dim=1)  # B, C, H, W
        return x 

    @torch.inference_mode()
    def sample(self, batch_size, temperature=1.): 
        logits, prior_zs = self.top_down.sample_from_prior(
            batch_size=batch_size, 
            temperature=temperature
        )

        return self._sample_from_logits(logits, temperature=temperature)
