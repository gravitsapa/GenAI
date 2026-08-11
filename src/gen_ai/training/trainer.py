from typing import Optional
from dataclasses import dataclass, asdict

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm, trange

from gen_ai.metadata.collectors import DeclarationMetadata, DeclarationDescribed

from gen_ai.project_config import EXPERIMENTS_DIR
from gen_ai.models.common import get_model_device
from gen_ai.models.generative import ImageGenerativeModel
from gen_ai.training.logger import Logger


@dataclass
class TrainerConfig:
    num_epochs: int
    experiment_name: str


class Trainer(DeclarationDescribed):
    def __init__(
        self,
        model: ImageGenerativeModel,
        data_loader: DataLoader,
        loss_function: nn.Module,
        optimizer: Optimizer,
        logger: Logger,
        scheduler: LRScheduler,
        config: TrainerConfig,
    ):
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.config = config


    def _train_one_epoch(self) -> float:
        loss_sum: float = 0.0
        loss_cnt: int = 0

        device = get_model_device(self.model)

        for sample in tqdm(self.data_loader, desc="Batch", leave=False):
            image = sample.image.to(device)

            self.optimizer.zero_grad()

            model_output = self.model(image)
            loss = self.loss_function(image, model_output)
            loss.backward()

            self.optimizer.step()

            loss_sum += float(loss.detach().cpu())
            loss_cnt += 1

        return loss_sum / loss_cnt


    def train_loop(
        self,
    ) -> list[float]:
        loss_history: list[float] = []

        self.model.train()
        for epoch_num in trange(1, self.config.num_epochs + 1, desc="Epoch"):
            loss = self._train_one_epoch()

            loss_history.append(loss)

            print(f"Epoch {epoch_num}. Loss = {loss:.6f}")

            # weights_filename = experiment_dir / f"weights_epoch_{epoch_num}.pth"
            # torch.save(self.model.state_dict(), weights_filename)

            self.scheduler.step()

        return loss_history

    def get_declaration_metadata(self) -> DeclarationMetadata:
        return DeclarationMetadata(asdict(self.config))
