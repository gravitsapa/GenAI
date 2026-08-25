from typing import Optional
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gen_ai.models.eff_vdvae.cnn import conv2d


class GaussianLatentLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        num_variates,
        gradient_smoothing_beta: float=np.log(2),
    ):
        super().__init__()

        self.projection = conv2d(
            in_channels=in_channels,
            out_channels=2 * num_variates,
            kernel_size=1,
        )

        self.softplus = torch.nn.Softplus(beta=gradient_smoothing_beta)

    def sample(
        self,
        dist_params: tuple[Tensor, Tensor],
        device: torch.device,
        temperature: float = 1.,
    ) -> Tensor:
        mean, std = dist_params

        eps = torch.randn_like(mean, device=device)
        return temperature * eps * std + mean

    def forward(
        self,
        input_tensor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x = input_tensor
        x = self.projection(x)

        mean, std = torch.chunk(x, chunks=2, dim=1)
        std = self.softplus(std)

        return mean, std

