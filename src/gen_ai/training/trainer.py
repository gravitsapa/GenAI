from typing import Optional
from dataclasses import dataclass, asdict

from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from gen_ai.metadata.collectors import DeclarationMetadata, DeclarationDescribed

from gen_ai.data.dataloader import DescribedImageDataLoader
from gen_ai.data.datasets import DescribedImageDataset
from gen_ai.training.optimizer import DescribedOptimizer
from gen_ai.training.scheduler import DescribedScheduler
from gen_ai.models.common import get_model_device
from gen_ai.models.generative import DescribedImageGenerativeModel
from gen_ai.training.logger import Logger


@dataclass
class TrainerConfig:
    num_epochs: int


class Trainer(DeclarationDescribed):
    def __init__(
        self,
        model: DescribedImageGenerativeModel,
        dataset: DescribedImageDataset,
        data_loader: DescribedImageDataLoader,
        loss_function: nn.Module,
        optimizer: DescribedOptimizer,
        logger: Logger,
        scheduler: DescribedScheduler,
        config: TrainerConfig,
    ):
        self.model = model
        self.dataset = dataset
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


    def _collect_config(self) -> dict:
        experiment_config = {}
        experiment_config['model_config'] = self.model.get_declaration_metadata_dict()
        experiment_config['dataset_config'] = self.dataset.get_declaration_metadata_dict()
        experiment_config['dataloader_config'] = self.data_loader.get_declaration_metadata_dict()
        experiment_config['optimizer_config'] = self.optimizer.get_declaration_metadata_dict()
        experiment_config['scheduler_config'] = self.scheduler.get_declaration_metadata_dict()
        experiment_config['trainer_config'] = self.get_declaration_metadata_dict()

        return experiment_config


    def train_loop(
        self,
    ) -> list[float]:
        self.logger.log_config(self._collect_config())
        
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

    def _get_specific_declaration_metadata(self) -> DeclarationMetadata:
        return DeclarationMetadata(asdict(self.config))
