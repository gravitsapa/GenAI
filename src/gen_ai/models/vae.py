import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass

from gen_ai.models.cnn import Encoder, Decoder


@dataclass
class VAEResult:
    latent: Tensor
    output_tensor: Tensor


class VAE(nn.Module):
    def __init__(
        self,
        hidden_channels: int=3,
        block_channels: tuple[int, ...]=(64,),
        mid_layers: int=2,
        norm_num_groups: int=8,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=3,
            out_channels=hidden_channels,
            block_channels=block_channels,
            mid_layers=mid_layers,
            norm_num_groups=norm_num_groups,
        )

        self.decoder = Decoder(
            in_channels=hidden_channels,
            out_channels=3,
            block_channels=block_channels,
            mid_layers=mid_layers,
            norm_num_groups=norm_num_groups,
        )


    def forward(self, input_tensor: Tensor) -> VAEResult:
        latent = self.encoder(input_tensor)
        output_tensor = self.decoder(latent)

        return VAEResult(latent, output_tensor)


    @torch.inference_mode()
    def sample(self, latent: Tensor) -> Tensor:
        output_tensor = self.decoder(latent)
        return output_tensor


class VAELoss(nn.Module):
    def __init__(self, beta: float=1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        input_tensor: Tensor,
        vae_result: VAEResult,
    ):
        reconstruction_loss = F.mse_loss(
            input_tensor,
            vae_result.output_tensor,
            reduction="sum",
        )

        latent_channels_x2 = int(vae_result.latent.shape[1])
        assert latent_channels_x2 % 2 == 0
        latent_channels = latent_channels_x2 // 2

        mu = vae_result.latent[:, :latent_channels, :, :]
        var = vae_result.latent[:, latent_channels:, :, :]

        kl_divergence = 0.5 * torch.sum(
            mu.square() + var - 1 - torch.log(var)
        )

        return reconstruction_loss + self.beta * kl_divergence


    