from gen_ai.data.datasets import AnimeFaces256
from gen_ai.data.image_sample import show_sample
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from gen_ai.models.vae import VAE, VAELoss
from gen_ai.training.trainer import Trainer

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Running on {device}")

    anime_faces = AnimeFaces256(64, 64)
    data_loader = DataLoader(
        anime_faces,
        32,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    print("Loaded dataset")

    vae = VAE(
        8,
        (64, 128, 256, 512),
        norm_num_groups=32,
    ).to(device)

    num_epochs = 200

    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    trainer = Trainer(
        vae,
        data_loader,
        VAELoss(),
        optimizer,
        "First attempt",
    )

    trainer.train_loop(num_epochs, scheduler)

    print("Done")
    
