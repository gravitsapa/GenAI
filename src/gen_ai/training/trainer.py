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
from gen_ai.training.metrics import WeightedMeanMetrics
from gen_ai.sampling.sampler import Sampler


@dataclass
class TrainerConfig:
    num_epochs: int
    log_every_epoch: int


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


    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()

        metrics_accumulator = WeightedMeanMetrics()

        device = get_model_device(self.model)

        for sample in tqdm(self.data_loader, desc="Batch", leave=False):
            image = sample.image.to(device)

            self.optimizer.zero_grad()

            model_output = self.model(image)
            loss, batch_loss_metrics = self.loss_function(image, model_output)

            loss.backward()

            self.optimizer.step()

            metric_values = dict(batch_loss_metrics.to_dict())
            if "loss" in metric_values:
                raise ValueError('"loss" is reserved for the optimized loss tensor')

            metric_values["loss"] = loss.detach().item()
            metrics_accumulator.update(metric_values, batch_size=image.shape[0])

        return metrics_accumulator.compute()


    def _collect_config(self) -> dict:
        experiment_config = {}
        experiment_config['model_config'] = self.model.get_declaration_metadata_dict()
        experiment_config['dataset_config'] = self.dataset.get_declaration_metadata_dict()
        experiment_config['dataloader_config'] = self.data_loader.get_declaration_metadata_dict()
        experiment_config['optimizer_config'] = self.optimizer.get_declaration_metadata_dict()
        experiment_config['scheduler_config'] = self.scheduler.get_declaration_metadata_dict()
        experiment_config['trainer_config'] = self.get_declaration_metadata_dict()

        return experiment_config


    def _collect_metrics(
        self,
        epoch_num: int,
        loss_metrics: dict[str, float],
    ) -> dict:
        return {
            "epoch": epoch_num,
            "learning_rates": [
                parameter_group["lr"]
                for parameter_group in self.optimizer.param_groups
            ],
            **loss_metrics,
        }


    def train_loop(
        self,
    ) -> list[dict]:
        self.logger.log_config(self._collect_config())

        sampler = Sampler(self.model)

        metrics_history: list[dict] = []

        self.model.train()
        for epoch_num in trange(1, self.config.num_epochs + 1, desc="Epoch"):
            epoch_metrics = self._train_one_epoch()

            metrics_history.append(epoch_metrics)

            self.logger.log_metrics(
                self._collect_metrics(epoch_num, epoch_metrics)
            )

            self.scheduler.step()

            if epoch_num % self.config.log_every_epoch == 0:
                self.logger.save_metrics_plot(
                    metrics_history,
                    epoch_num,
                )

                checkpoint = {
                    "epoch": epoch_num,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "metrics_history": metrics_history,
                }

                self.logger.save_checkpoint(
                    checkpoint,
                    epoch_num,
                )

                samples_fig = sampler.sample_grid()
                self.logger.save_samples(
                    samples_fig,
                    epoch_num,
                )

        return metrics_history

    def _get_specific_declaration_metadata(self) -> DeclarationMetadata:
        return DeclarationMetadata(asdict(self.config))
