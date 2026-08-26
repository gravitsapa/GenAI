import numpy as np
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.metadata.configuration_collector import ContainingConfiguration
from gen_ai.exceptions import ConfigurationError, ModelShapeError, require
from gen_ai.models.common import get_model_device, one_hot
from gen_ai.models.generative import DescribedImageGenerativeModel, ImageShape
from gen_ai.models.eff_vdvae.loss import logistic_mixture_log_tensor, diag_normal_kl_divergence
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
from gen_ai.training.param_scheduler import ScheduledParam, get_param


@dataclass(frozen=True, kw_only=True)
class EffVDVAEConfig:
    image_shape: ImageShape
    n_layers_in_block: tuple[int, ...]
    blocks_channels_bottom_up: tuple[int, ...]
    blocks_strides_bottom_up: tuple[int, ...]
    blocks_skip_channels: tuple[int, ...]
    blocks_latent_variates: tuple[int, ...]
    n_output_mixtures: int
    input_min_value: float = -1.
    input_max_value: float = 1.
    in_channels: int = 3
    in_kernel: int = 1
    out_kernel: int = 1
    n_residual_conv_cells_in_layer: int = 1
    n_conv_layers_in_residual: int = 2
    kernel_size: int = 3
    bottleneck_channels_ratio: float = 0.25
    smoothing_beta: float = np.log(2)
    min_scale: float = np.exp(-250)

    def __post_init__(self) -> None:
        require(
            len(self.image_shape) == 2 and all(size > 0 for size in self.image_shape),
            ConfigurationError,
            f"image_shape must contain two positive dimensions, got {self.image_shape}",
        )

        block_counts = (
            len(self.n_layers_in_block),
            len(self.blocks_channels_bottom_up),
            len(self.blocks_strides_bottom_up),
            len(self.blocks_skip_channels),
            len(self.blocks_latent_variates),
        )
        require(
            block_counts[0] > 0,
            ConfigurationError,
            "blocks_channels_bottom_up must not be empty",
        )
        require(
            len(set(block_counts)) == 1,
            ConfigurationError,
            lambda: (
                "blocks_channels_bottom_up, blocks_strides_bottom_up, "
                "blocks_skip_channels, and blocks_latent_variates must have the same length, "
                f"got {block_counts}"
            ),
        )
        require(
            all(n_layers >= 0 for n_layers in self.n_layers_in_block),
            ConfigurationError,
            "n_layers_in_block must contain non negative layer counts",
        )
        require(
            all(channels > 0 for channels in self.blocks_channels_bottom_up),
            ConfigurationError,
            "blocks_channels_bottom_up must contain positive channel counts",
        )
        require(
            all(stride > 0 for stride in self.blocks_strides_bottom_up),
            ConfigurationError,
            "blocks_strides_bottom_up must contain positive strides",
        )
        require(
            all(channels > 0 for channels in self.blocks_skip_channels),
            ConfigurationError,
            "blocks_skip_channels must contain positive channel counts",
        )
        require(
            all(variates > 0 for variates in self.blocks_latent_variates),
            ConfigurationError,
            "blocks_latent_variates must contain positive values",
        )

        require(
            self.in_channels == 3,
            ConfigurationError,
            f"only RGB input is supported, got in_channels={self.in_channels}",
        )
        require(
            np.isclose(self.input_min_value, -1.).item()
            and np.isclose(self.input_max_value, 1.).item(),
            ConfigurationError,
            "only image values in [-1, 1] are supported",
        )
        require(
            self.n_output_mixtures > 0,
            ConfigurationError,
            f"n_output_mixtures must be positive, got {self.n_output_mixtures}",
        )
        require(
            self.n_residual_conv_cells_in_layer >= 0,
            ConfigurationError,
            "n_residual_conv_cells_in_layer must be non-negative, "
            f"got {self.n_residual_conv_cells_in_layer}",
        )
        require(
            self.n_conv_layers_in_residual >= 0,
            ConfigurationError,
            f"n_conv_layers_in_residual must be non-negative, got {self.n_conv_layers_in_residual}",
        )
        require(
            self.in_kernel > 0 and self.out_kernel > 0 and self.kernel_size > 0,
            ConfigurationError,
            "in_kernel, out_kernel, and kernel_size must be positive",
        )
        require(
            self.bottleneck_channels_ratio > 0,
            ConfigurationError,
            "bottleneck_channels_ratio must be positive, "
            f"got {self.bottleneck_channels_ratio}",
        )

        downsampling_factor = int(np.prod(self.blocks_strides_bottom_up))
        require(
            all(size % downsampling_factor == 0 for size in self.image_shape),
            ConfigurationError,
            lambda: (
                f"image_shape {self.image_shape} must be divisible by the downsampling "
                f"factor {downsampling_factor}"
            ),
        )

        residual_bottlenecks = [
            int(channels * self.bottleneck_channels_ratio)
            for channels in self.blocks_channels_bottom_up
        ]
        posterior_bottlenecks = [
            int(
                (channels + skip_channels)
                * 0.5
                * self.bottleneck_channels_ratio
            )
            for channels, skip_channels in zip(
                self.blocks_channels_bottom_up,
                self.blocks_skip_channels,
                strict=True,
            )
        ]
        require(
            all(channels > 0 for channels in residual_bottlenecks + posterior_bottlenecks),
            ConfigurationError,
            lambda: (
                "bottleneck_channels_ratio is too small for the configured channel counts: "
                "every residual bottleneck must contain at least one channel"
            ),
        )


class EffVDVAE(DescribedImageGenerativeModel):
    def __init__(
        self,
        config: EffVDVAEConfig,
    ):
        super().__init__(config=config)

        self.config = config
        layers_cnt = sum(n_layers + 1 for n_layers in self.config.n_layers_in_block)

        init_scaler = np.sqrt(1. / float(layers_cnt))

        bottom_up_common_config = BottomUpBlocksCommon(
            blocks_config=BottomUpBlocksConfig(
                n_residual_conv_cells=self.config.n_residual_conv_cells_in_layer,
                res_conv_common=ResConvCellCommonInBlocksUp(
                    kernel_size=self.config.kernel_size,
                    n_layers=self.config.n_conv_layers_in_residual,
                    init_scaler=init_scaler,
                    bottleneck_channels_ratio=self.config.bottleneck_channels_ratio,
                )
            )
        )

        self.bottom_up = BottomUp(
            blocks_common_config=bottom_up_common_config,
            n_layers_in_block=self.config.n_layers_in_block,
            blocks_channels=self.config.blocks_channels_bottom_up,
            blocks_stride=self.config.blocks_strides_bottom_up,
            blocks_skip_channels=self.config.blocks_skip_channels,
            in_channels=self.config.in_channels,
            in_conv_kernel=self.config.in_kernel,
        )

        top_down_common_config = TopDownBlocksCommon(
            blocks_config=TopDownBlocksConfig(
                n_residual_conv_cells=self.config.n_residual_conv_cells_in_layer,
                res_conv_common=ResConvCellCommonInBlockDown(
                    kernel_size=self.config.kernel_size,
                    n_layers=self.config.n_conv_layers_in_residual,
                    init_scaler=init_scaler,
                    bottleneck_channels_ratio=self.config.bottleneck_channels_ratio,
                ),
            )
        )

        out_channels = self.config.n_output_mixtures * (self.config.in_channels * 3 + 1)

        self.top_down = TopDown(
            image_shape=self.config.image_shape,
            blocks_common_config=top_down_common_config,
            n_layers_in_block=self.config.n_layers_in_block[::-1],
            blocks_channels=self.config.blocks_channels_bottom_up[::-1],
            blocks_stride=self.config.blocks_strides_bottom_up[::-1],
            blocks_skip_channels=self.config.blocks_skip_channels[::-1],
            blocks_latent_variates=self.config.blocks_latent_variates[::-1],
            out_channels=out_channels,
            out_conv_kernel=self.config.out_kernel,
        )

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]:
        require(
            input_tensor.ndim == 4
            and input_tensor.shape[1] == self.config.in_channels
            and tuple(input_tensor.shape[-2:]) == self.config.image_shape,
            ModelShapeError,
            lambda: (
                f"expected input shape (B, {self.config.in_channels}, "
                f"{self.config.image_shape[0]}, {self.config.image_shape[1]}), "
                f"got {tuple(input_tensor.shape)}"
            ),
        )
        skip_list = self.bottom_up(input_tensor)
        logits, posterior_params_list, prior_params_list = self.top_down(skip_list[::-1])

        return logits, posterior_params_list, prior_params_list


    def _parse_logits(
        self,
        logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = get_model_device(self)

        n_mixtures = self.config.n_output_mixtures
        in_channels = self.config.in_channels

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
            (H, W) == self.config.image_shape,
            ValueError,
            lambda: (
                f"Invalid logits spatial shape: expected {self.config.image_shape}, got {(H, W)}"
            ),
        )

        logit_probs = logits[:, :n_mixtures, :, :]  # B, M, H, W
        l = logits[:, n_mixtures:, :, :]  # B, M*C*3 ,H, W
        l = l.reshape(B, in_channels, 3 * n_mixtures, H, W)  # B, C, 3 * M, H, W

        model_means = l[:, :, :n_mixtures, :, :]  # B, C, M, H, W
        scales_logits = l[:, :, n_mixtures: 2 * n_mixtures, :, :] # B, C, M, H, W
        model_coeffs = torch.tanh(
            l[:, :, 2 * n_mixtures: 3 * n_mixtures, :, :]
        )  # B, C, M, H, W

        softplus = nn.Softplus(beta=self.config.smoothing_beta)
        scales = torch.maximum(
            softplus(scales_logits), 
            torch.as_tensor(self.config.min_scale, device=device),
        )

        return logit_probs, model_means, scales, model_coeffs

    
    @torch.inference_mode()
    def _sample_from_logits(
        self,
        logits: Tensor,
        temperature: float = 1.,
    ) -> Tensor:
        device = get_model_device(self)
        
        n_mixtures = self.config.n_output_mixtures

        logit_probs, model_means, scales, model_coeffs = self._parse_logits(logits)

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
            min=self.config.input_min_value,
            max=self.config.input_max_value
        )  # B, 1, H, W
        x1 = torch.clamp(
            x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0, 
            min=self.config.input_min_value,
            max=self.config.input_max_value
        )  # B, 1, H, W
        x2 = torch.clamp(
            x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1,
            min=self.config.input_min_value,
            max=self.config.input_max_value
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


@dataclass(frozen=True, kw_only=True)
class EffVDVAELossConfig:
    beta: float | ScheduledParam[float]


@dataclass
class EffVDVAELossData:
    reconstruction_loss: float
    kl_divergence: float
    beta: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class EffVDVAELoss(ContainingConfiguration, nn.Module):
    def __init__(
        self,
        model: EffVDVAE,
        config: EffVDVAELossConfig,
    ):
        super().__init__(config=config)
        self.model = model

    def forward(
        self,
        input_tensor: Tensor,
        forward_output: tuple[Tensor, list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]],
    ) -> tuple[Tensor, EffVDVAELossData]:
        logits, posterior_params_list, prior_params_list = forward_output

        logit_probs, model_means, scales, model_coeffs = self.model._parse_logits(logits)
        log_pi = F.log_softmax(logit_probs, dim=1)

        means0 = model_means[:, 0, :, :, :]
        means1 = (
            model_means[:, 1, :, :, :] +
            input_tensor[:, 0:1, :, :] * model_coeffs[:, 0, :, :, :]
        )
        means2 = (
            model_means[:, 2, :, :, :] +
            input_tensor[:, 0:1, :, :] * model_coeffs[:, 1, :, :, :] +
            input_tensor[:, 1:2, :, :] * model_coeffs[:, 2, :, :, :]
        )
        
        scalar = input_tensor.numel()

        nll = -torch.sum(torch.logsumexp(
            log_pi +
            logistic_mixture_log_tensor(
                input_tensor=input_tensor[:, 0, :, :],
                means=means0,
                scales=scales[:, 0, :, :, :],
            ) + 
            logistic_mixture_log_tensor(
                input_tensor=input_tensor[:, 1, :, :],
                means=means1,
                scales=scales[:, 1, :, :, :],
            ) + 
            logistic_mixture_log_tensor(
                input_tensor=input_tensor[:, 2, :, :],
                means=means2,
                scales=scales[:, 2, :, :, :],
            ),
            dim=1
        )) / scalar

        kl_div_list = torch.stack([
            diag_normal_kl_divergence(q, p)
            for q, p in zip(
                posterior_params_list,
                prior_params_list,
                strict=True
            )
        ], dim=0)

        kl_div = torch.sum(kl_div_list) / scalar

        beta = get_param(self.config.beta)
        loss = nll + beta * kl_div

        return loss, EffVDVAELossData(
            reconstruction_loss=nll.detach().cpu().item(),
            kl_divergence=kl_div.detach().cpu().item(),
            beta=beta,
        )
