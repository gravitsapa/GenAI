from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.models.common import ModuleFactory, get_model_device
from gen_ai.models.eff_vdvae.cnn import conv2d, ResidualConvCell
from gen_ai.models.eff_vdvae.latent_layers import GaussianLatentLayer


class Interpolate(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        non_linearity: ModuleFactory=lambda: nn.LeakyReLU(negative_slope=0.1),
    ):
        super().__init__()

        self.ops = nn.Sequential(*[
            conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            non_linearity(),
            Interpolate(stride),
            conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        ])

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.ops(input_tensor)


@dataclass(kw_only=True)
class ResConvCellCommonInBlockDown:
    kernel_size: int | tuple[int, int]
    n_layers: int
    init_scaler: float
    bottleneck_channels_ratio: float = 0.25


class BlockDown(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_residual_conv_cells: int,
        residual_conv_cell_config: ResConvCellCommonInBlockDown,
        skip_channels: int,
        latent_variates: int,
        out_channels: Optional[int]=None,
        stride: int = 1,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if stride > 1: 
            self.resample = Upsample(
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

        self.residual_block = nn.Sequential(*[
            ResidualConvCell(
                kernel_size=residual_conv_cell_config.kernel_size,
                n_layers=residual_conv_cell_config.n_layers,
                bottleneck_channels_ratio=residual_conv_cell_config.bottleneck_channels_ratio,
                init_scaler=residual_conv_cell_config.init_scaler,
                in_channels=out_channels,
                out_channels=out_channels,
            )
            for _ in range(n_residual_conv_cells)
        ])

        self.posterior_net = ResidualConvCell(
            kernel_size=residual_conv_cell_config.kernel_size,
            n_layers=residual_conv_cell_config.n_layers,
            bottleneck_channels_ratio=0.5 * residual_conv_cell_config.bottleneck_channels_ratio,
            init_scaler=1.,
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

        self.prior_net = ResidualConvCell(
            kernel_size=residual_conv_cell_config.kernel_size,
            n_layers=residual_conv_cell_config.n_layers,
            bottleneck_channels_ratio=residual_conv_cell_config.bottleneck_channels_ratio,
            init_scaler=1.,
            in_channels=out_channels,
            out_channels=2 * out_channels,  
        )

        self.prior_layer = GaussianLatentLayer(
            in_channels=out_channels,
            num_variates=latent_variates,
        )

        self.posterior_layer = GaussianLatentLayer(
            in_channels=out_channels,
            num_variates=latent_variates,
        )

        self.z_projection = conv2d(
            in_channels=latent_variates,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.z_projection.weight.data *= residual_conv_cell_config.init_scaler

    def forward(
        self,
        y: Tensor,
        x_skip: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        device = get_model_device(self)

        y = self.resample(y)

        prior_residual, prior_dist_params_input = torch.chunk(
            self.prior_net(y),
            chunks=2,
            dim=1,
        )

        prior_params = self.prior_layer(prior_dist_params_input)

        y_posterior = self.posterior_net(
            torch.cat([y, x_skip], dim=1)
        )

        posterior_params = self.posterior_layer(y_posterior)
        z_posterior = self.posterior_layer.sample(
            posterior_params,
            device=device,
        )

        y = y + prior_residual

        z_post_projection = self.z_projection(z_posterior)
        y = y + z_post_projection

        y = self.residual_block(y)

        return y, posterior_params, prior_params
