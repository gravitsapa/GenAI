import numpy as np
import torch

from gen_ai.data.augmentations import AugmentationsConfig, AugmentationBuilder
from gen_ai.data.datasets import AnimeFaces256, ImageDatasetConfig
from gen_ai.data.dataloader import DescribedImageDataLoader, DataloaderConfig
from gen_ai.data.fetcher import DataFetcher
from gen_ai.models.eff_vdvae.eff_vdvae import (
    EffVDVAE, EffVDVAEConfig, 
    EffVDVAELoss, EffVDVAELossConfig
)
from gen_ai.training.optimizer import DescribedAdamax, AdamaxConfig
from gen_ai.training.scheduler import (
    DescribedCosineAnnealingLR,
    CosineAnnealingLRConfig,
    DescribedSequentialLR,
    DescribedLinearLR,
    LinearLRConfig,
    DescribedConstantLR,
    ConstantLRConfig,
)
from gen_ai.training.trainer import Trainer, TrainerConfig
from gen_ai.training.logger import Logger
from gen_ai.training.param_scheduler import (
    ParamScheduler, 
    ScheduledParam, 
    LinearFloatScheme, 
    LinearFloatSchemeConfig
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Running on {device}")

    image_shape = (64, 64)
    batch_size = 8
    num_steps = 1600000
    warmup_steps = 2000
    beta_warmup_steps = 10000
    decay_begin_step = 100000

    augmentation_builder = AugmentationBuilder(AugmentationsConfig(
        random_crop_scale=None,
        random_horizontal_flip=0.5,
    ))

    anime_faces = AnimeFaces256(
        ImageDatasetConfig(
            image_shape=image_shape,
        ),
        augmentation_builder=augmentation_builder,
    )
    data_loader = DescribedImageDataLoader(
        anime_faces,
        DataloaderConfig(
            batch_size=batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
            num_workers=2,
            persistent_workers=True,
            drop_last=True,
        )
    )
    data_fetcher = DataFetcher(data_loader)

    print("Loaded dataset")

    eff_vdvae = EffVDVAE(EffVDVAEConfig(
        image_shape=image_shape,
        n_layers_in_block=(2,) * 15 + (1,) * 7,
        blocks_channels_bottom_up=(32,)*6 + (64,)*5 + (128,)*4 + (256,)*4 + (512,)*3,
        blocks_strides_bottom_up=(1,) * 5 + (2,) + (1,) * 4 + (2,) + (1,) * 3 + (2,) + (1,) * 3 + (2,) + (1, 4) + (1,),
        blocks_skip_channels=(32,)*6 + (64,)*5 + (128,)*4 + (256,)*4 + (512,)*3,
        blocks_latent_variates=(32,) * 22,
        n_output_mixtures=10,
        n_residual_conv_cells_in_layer=1,
        n_conv_layers_in_residual=2,
        min_scale=np.exp(-10),
    )).to(device)   

    optimizer = DescribedAdamax(
        eff_vdvae.parameters(),
        AdamaxConfig(
            lr=1e-3,
        )
    )

    scheduler = DescribedSequentialLR(
        optimizer=optimizer,
        schedulers=[
            DescribedLinearLR(
                optimizer,
                LinearLRConfig(
                    total_iters=warmup_steps,
                )
            ),
            DescribedConstantLR(
                optimizer,
                ConstantLRConfig(
                    factor=1.,
                    total_iters=decay_begin_step - warmup_steps,
                )
            ),
            DescribedCosineAnnealingLR(
                optimizer,
                CosineAnnealingLRConfig(
                    T_max=num_steps - decay_begin_step,
                    eta_min=1e-4,
                ),
            )
        ],
        milestones=[
            warmup_steps,
            decay_begin_step,
        ]
    )

    logger = Logger("eff_vdvae_64_2.2")

    param_scheduler = ParamScheduler()

    loss = EffVDVAELoss(
        model=eff_vdvae,
        config=EffVDVAELossConfig(
            beta=ScheduledParam(
                param_scheduler=param_scheduler,
                scheme=LinearFloatScheme(
                    LinearFloatSchemeConfig(
                        begin_step=1,
                        end_step=beta_warmup_steps,
                        begin_value=1e-4,
                        end_value=1.,
                    )
                )
            )
        )
    )

    trainer = Trainer(
        eff_vdvae,
        data_fetcher,
        loss,
        optimizer,
        logger,
        scheduler,
        param_scheduler,
        TrainerConfig(
            num_steps=num_steps,
            eval_and_save_every_step=num_steps // 100,
            collect_metrics_every_step=num_steps // 10000,
            gradient_skip_threshold=800,
        ),
    )

    trainer.train_loop()

    print("Done")
    