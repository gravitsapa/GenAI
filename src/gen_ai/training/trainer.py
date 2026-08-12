from typing import Optional
from dataclasses import dataclass, asdict

from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from gen_ai.metadata.configuration_collector import ContainingConfiguration

from gen_ai.data.dataloader import DescribedImageDataLoader
from gen_ai.data.datasets import DescribedImageDataset
from gen_ai.training.optimizer import DescribedOptimizer
from gen_ai.training.scheduler import DescribedScheduler
from gen_ai.models.common import get_model_device
from gen_ai.models.generative import DescribedImageGenerativeModel
from gen_ai.training.logger import Logger
from gen_ai.training.metrics import WeightedMeanMetrics
from gen_ai.sampling.sampler import Sampler
from gen_ai.exceptions import ConfigurationError, TrainingError, require


@dataclass
class TrainerConfig:
    num_epochs: int
    log_every_epoch: int

    def __post_init__(self) -> None:
        require(
            self.num_epochs > 0,
            ConfigurationError,
            f"num_epochs must be positive, got {self.num_epochs}",
        )
        require(
            self.log_every_epoch > 0,
            ConfigurationError,
            f"log_every_epoch must be positive, got {self.log_every_epoch}",
        )


class Trainer(ContainingConfiguration):
    def __init__(
        self,
        model: DescribedImageGenerativeModel,
        data_loader: DescribedImageDataLoader,
        loss_function: nn.Module,
        optimizer: DescribedOptimizer,
        logger: Logger,
        scheduler: DescribedScheduler,
        config: TrainerConfig,
    ):
        super().__init__(config=config)

        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler


    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()

        metrics_accumulator = WeightedMeanMetrics()

        device = get_model_device(self.model)

        for sample in tqdm(self.data_loader, desc="Batch", leave=False):
            image = sample.image.to(device)

            self.optimizer.zero_grad()

            model_output = self.model(image)
            loss, batch_loss_metrics = self.loss_function(image, model_output)

            require(
                loss.ndim == 0,
                TrainingError,
                f"loss function must return a scalar tensor, got shape {tuple(loss.shape)}",
            )

            loss.backward()

            self.optimizer.step()

            metric_values = dict(batch_loss_metrics.to_dict())
            if "loss" in metric_values:
                raise ValueError('"loss" is reserved for the optimized loss tensor')

            metric_values["loss"] = loss.detach().item()
            metrics_accumulator.update(metric_values, batch_size=image.shape[0])

        return metrics_accumulator.compute()


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
        require(
            len(self.data_loader) > 0,
            TrainingError,
            "cannot train with an empty data loader",
        )
        self.logger.log_config(self.get_metadata_dict())

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

