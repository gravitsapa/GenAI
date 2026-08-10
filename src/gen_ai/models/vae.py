import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass

from gen_ai.models.cnn import Encoder, Decoder


def split_channels_on_2_parts(tensor: Tensor) -> tuple[Tensor, Tensor]:
    channels_x2 = int(tensor.shape[1])
    assert channels_x2 % 2 == 0
    channels = channels_x2 // 2

    left = tensor[:, :channels, :, :]
    right = tensor[:, channels:, :, :]

    return left, right


@dataclass
class VAEResult:
    encoder_output: Tensor
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
            double_output=True
        )

        self.decoder = Decoder(
            in_channels=hidden_channels,
            out_channels=3,
            block_channels=block_channels,
            mid_layers=mid_layers,
            norm_num_groups=norm_num_groups,
        )

        self.head = nn.Tanh()


    def forward(self, input_tensor: Tensor) -> VAEResult:
        encoder_output = self.encoder(input_tensor)

        mu, log_var = split_channels_on_2_parts(encoder_output)
        std = torch.exp(0.5 * log_var)

        eps = torch.randn_like(mu)
        latent = mu + std * eps

        decoder_output = self.decoder(latent)
        output_tensor = self.head(decoder_output)

        return VAEResult(encoder_output, output_tensor)


    @torch.inference_mode()
    def sample(self, latent: Tensor) -> Tensor:
        decoder_output = self.decoder(latent)
        output_tensor = self.head(decoder_output)
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
            reduction="none",
        ).flatten(1).sum(dim=1)

        mu, log_var = split_channels_on_2_parts(vae_result.encoder_output)
        var = torch.exp(log_var)

        kl_divergence = 0.5 * (
            mu.square() + var - 1 - log_var
        ).flatten(1).sum(dim=1)

        return (reconstruction_loss + self.beta * kl_divergence).mean()


    
