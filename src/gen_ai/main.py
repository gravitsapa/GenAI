import torch

from gen_ai.data.augmentations import AugmentationsConfig, AugmentationBuilder
from gen_ai.data.datasets import AnimeFaces256, ImageDatasetConfig
from gen_ai.data.dataloader import DescribedImageDataLoader, DataloaderConfig
from gen_ai.models.vae.vae import VAE, VAELoss, VAEConfig
from gen_ai.training.optimizer import DescribedAdamW, AdamWConfig
from gen_ai.training.scheduler import DescribedCosineAnnealingLR, CosineAnnealingLRConfig, DescribedSequentialLR, DescribedLinearLR, LinearLRConfig
from gen_ai.training.trainer import Trainer, TrainerConfig
from gen_ai.training.logger import Logger

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Running on {device}")

    image_shape = (64, 64)
    num_epochs=200

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
            batch_size=48,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True,
        )
    )

    print("Loaded dataset")

    vae = VAE(VAEConfig(
        image_shape=image_shape,
        image_channels=3,
        hidden_channels=4,
        block_channels=(64, 128, 256, 512),
        norm_num_groups=32,
        mid_layers=2,
    )).to(device)

    checkpoint_filename = "D:\\Documents\\GenAI\\experiments\\extended_vae_wo_augment_batch48_lr2e-4_2026-08-14_00-29\\checkpoints\\checkpoint_epoch_0200.pt"
    checkpoint_file = torch.load(checkpoint_filename, weights_only=False, map_location=device)

    vae.load_state_dict(checkpoint_file['model_state_dict'])
    print("Successfully loaded weights")

    optimizer = DescribedAdamW(
        vae.parameters(),
        AdamWConfig(
            lr=1e-4,
        )
    )

    scheduler = DescribedSequentialLR(
        optimizer=optimizer,
        schedulers=[
            DescribedLinearLR(
                optimizer,
                LinearLRConfig(
                    total_iters=20,
                )
            ),
            DescribedCosineAnnealingLR(
                optimizer,
                CosineAnnealingLRConfig(
                    T_max=num_epochs,
                    eta_min=1e-6,
                ),
            )
        ],
        milestones=[
            20,
        ]
    )


    logger = Logger("extended_vae_wo_augment_batch48_lr2e-4_epochs201-400")

    trainer = Trainer(
        vae,
        data_loader,
        VAELoss(),
        optimizer,
        logger,
        scheduler,
        TrainerConfig(
            num_epochs=num_epochs,
            log_every_epoch=20
        ),
    )

    trainer.train_loop()

    print("Done")
    
