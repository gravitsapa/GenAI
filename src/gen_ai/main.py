import torch

from gen_ai.data.datasets import AnimeFaces256, ImageDatasetConfig
from gen_ai.data.dataloader import DescribedImageDataLoader, DataloaderConfig
from gen_ai.models.vae import VAE, VAELoss, VAEConfig
from gen_ai.training.optimizer import DescribedAdam, AdamConfig
from gen_ai.training.scheduler import DescribedCosineAnnealingLR, CosineAnnealingLRConfig
from gen_ai.training.trainer import Trainer, TrainerConfig
from gen_ai.training.logger import Logger

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Running on {device}")

    image_shape = (64, 64)
    num_epochs=200

    anime_faces = AnimeFaces256(ImageDatasetConfig(
        image_shape=image_shape
    ))
    data_loader = DescribedImageDataLoader(
        anime_faces,
        DataloaderConfig(
            batch_size=32,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
    )

    print("Loaded dataset")

    vae = VAE(VAEConfig(
        image_shape=image_shape,
        image_channels=3,
        hidden_channels=8,
        block_channels=(64, 128, 256, 512),
        norm_num_groups=32,
        mid_layers=2
    )).to(device)


    optimizer = DescribedAdam(
        vae.parameters(),
        AdamConfig(
            lr=1e-4,
        )
    )

    scheduler = DescribedCosineAnnealingLR(
        optimizer,
        CosineAnnealingLRConfig(
            T_max=num_epochs,
            eta_min=1e-6,
        ),
    )

    logger = Logger("second_attempt")

    trainer = Trainer(
        vae,
        anime_faces,
        data_loader,
        VAELoss(),
        optimizer,
        logger,
        scheduler,
        TrainerConfig(
            num_epochs=num_epochs,
        ),
    )

    trainer.train_loop()

    print("Done")
    
