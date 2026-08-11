from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass

from gen_ai.models.common import ModuleFactory, get_model_device
from gen_ai.models.cnn import conv1x1, conv3x3, ResNetBlock2D, ResChange
from gen_ai.models.normalization import NormalizationFactory, GroupNormalizationFactory

from gen_ai.models.generative import ImageGenerativeModel


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        out_channels: int=3,
        block_channels: tuple[int, ...]=(64,),
        norm_num_groups: int=8,
        activation_fn: ModuleFactory=nn.SiLU,
        mid_layers: int=2,
        double_output: bool=False,
    ):
        super().__init__()

        self.conv_in = conv3x3(
            in_channels,
            block_channels[0]
        )

        self.down_blocks = nn.Sequential()
        last_channels = block_channels[0]
        
        for block_num, channels in enumerate(block_channels):
            block_in_channels = last_channels
            block_out_channels = channels

            last_channels = block_out_channels
            is_last_block = block_num == len(block_channels) - 1

            down_block = ResNetBlock2D(
                block_in_channels,
                block_out_channels,
                ResChange.IDENTIAL if is_last_block else ResChange.DOWN,
                non_linearity=activation_fn,
                normalization=GroupNormalizationFactory(norm_num_groups),
            )

            self.down_blocks.append(down_block)

        self.mid_blocks = nn.Sequential()
        for block_num in range(mid_layers):
            self.mid_blocks.append(ResNetBlock2D(
                in_channels=last_channels,
                out_channels=last_channels,
                res_change=ResChange.IDENTIAL,
                non_linearity=activation_fn,
                normalization=GroupNormalizationFactory(norm_num_groups),
            ))

        self.conv_norm_out = GroupNormalizationFactory(norm_num_groups)(last_channels)
        self.conv_act = activation_fn()
        
        conv_out_channels = 2 * out_channels if double_output else out_channels
        self.conv_out = conv3x3(last_channels, conv_out_channels)


    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = input_tensor

        hidden = self.conv_in(hidden)
        hidden = self.down_blocks(hidden)

        hidden = self.mid_blocks(hidden)

        hidden = self.conv_norm_out(hidden)
        hidden = self.conv_act(hidden)
        output_tensor = self.conv_out(hidden)

        return output_tensor


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        out_channels: int=3,
        block_channels: tuple[int, ...]=(64,),
        norm_num_groups: int=8,
        activation_fn: ModuleFactory=nn.SiLU,
        mid_layers: int=2,
    ):
        super().__init__()

        self.conv_in = conv3x3(
            in_channels,
            block_channels[-1]
        )

        last_channels = block_channels[-1]

        self.mid_blocks = nn.Sequential()
        for block_num in range(mid_layers):
            self.mid_blocks.append(ResNetBlock2D(
                in_channels=last_channels,
                out_channels=last_channels,
                res_change=ResChange.IDENTIAL,
                non_linearity=activation_fn,
                normalization=GroupNormalizationFactory(norm_num_groups),
            ))

        self.up_blocks = nn.Sequential()

        for block_num, channels in enumerate(reversed(block_channels)):
            block_in_channels = last_channels
            block_out_channels = channels

            last_channels = block_out_channels
            is_last_block = block_num == len(block_channels) - 1

            up_block = ResNetBlock2D(
                block_in_channels,
                block_out_channels,
                ResChange.IDENTIAL if is_last_block else ResChange.UP,
                non_linearity=activation_fn,
                normalization=GroupNormalizationFactory(norm_num_groups),
            )

            self.up_blocks.append(up_block)

        self.conv_norm_out = GroupNormalizationFactory(norm_num_groups)(last_channels)
        self.conv_act = activation_fn()

        self.conv_out = conv3x3(last_channels, out_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden = input_tensor

        hidden = self.conv_in(hidden)
        hidden = self.mid_blocks(hidden)
        hidden = self.up_blocks(hidden)

        hidden = self.conv_norm_out(hidden)
        hidden = self.conv_act(hidden)
        ouput_tensor = self.conv_out(hidden)

        return ouput_tensor
        

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


class VAE(ImageGenerativeModel):
    def __init__(
        self,
        image_shape: tuple[int, int],
        image_channels: int=3,
        hidden_channels: int=3,
        block_channels: tuple[int, ...]=(64,),
        mid_layers: int=2,
        norm_num_groups: int=8,
    ):
        super().__init__()

        self.image_shape = image_shape
        self.image_channels = image_channels
        self.hidden_channels = hidden_channels
        self.block_channels = block_channels
        self.mid_layers = mid_layers
        self.norm_num_groups = norm_num_groups

        self.encoder = Encoder(
            in_channels=image_channels,
            out_channels=hidden_channels,
            block_channels=block_channels,
            mid_layers=mid_layers,
            norm_num_groups=norm_num_groups,
            double_output=True
        )

        self.decoder = Decoder(
            in_channels=hidden_channels,
            out_channels=image_channels,
            block_channels=block_channels,
            mid_layers=mid_layers,
            norm_num_groups=norm_num_groups,
        )

        self.head = nn.Tanh()


    def _get_latent_shape(self, image_shape: tuple[int, int]) -> tuple[int, int, int]:
        divider = 2 ** (len(self.block_channels) - 1)

        assert image_shape[0] % divider == 0 and image_shape[1] % divider == 0
        return (self.hidden_channels, image_shape[0] // divider, image_shape[1] // divider)


    def _reparametrization(self, mu: Tensor, std: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        latent = mu + std * eps

        return latent


    def _gen_latent(self, batch_size: int, image_shape: tuple[int, int]) -> Tensor:
        shape = (batch_size, ) + self._get_latent_shape(image_shape)
        device = get_model_device(self)

        mu = torch.zeros(shape, device=device)
        std = torch.ones(shape, device=device)

        return self._reparametrization(mu, std)
    

    def _sample_by_latent(self, latent: Tensor) -> Tensor:
        decoder_output = self.decoder(latent)
        output_tensor = self.head(decoder_output)
        return output_tensor


    def forward(self, input_tensor: Tensor) -> VAEResult:
        encoder_output = self.encoder(input_tensor)

        mu, log_var = split_channels_on_2_parts(encoder_output)
        std = torch.exp(0.5 * log_var)

        latent = self._reparametrization(mu, std)

        decoder_output = self.decoder(latent)
        output_tensor = self.head(decoder_output)

        return VAEResult(encoder_output, output_tensor)


    def sample(self, batch_size: int) -> Tensor:
        latent = self._gen_latent(batch_size, self.image_shape)

        output_tensor = self._sample_by_latent(latent)
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
    
# Sources:
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py
