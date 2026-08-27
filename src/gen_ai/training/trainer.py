from collections.abc import Callable
from typing import Any, Optional
from dataclasses import dataclass, asdict

from pathlib import Path

import torch
from tqdm.auto import tqdm, trange

from gen_ai.metadata.configuration_collector import ContainingConfiguration

from gen_ai.data.dataloader import DescribedImageDataLoader
from gen_ai.data.datasets import DescribedImageDataset
from gen_ai.training.optimizer import DescribedOptimizer
from gen_ai.training.scheduler import DescribedScheduler
from gen_ai.training.param_scheduler import ParamScheduler
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
    gradient_skip_threshold: float
    epoch_bar_info_every_batch: int

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
        loss_function: Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]],
        optimizer: DescribedOptimizer,
        logger: Logger,
        scheduler: DescribedScheduler,
        param_scheduler: ParamScheduler,
        config: TrainerConfig,
    ):
        super().__init__(config=config)

        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.param_scheduler = param_scheduler

    def _skip_gradient_correctness(self) -> bool:
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        if len(parameters) == 0:
            gradient_norm = torch.tensor(0.0)
        else:
            device = parameters[0].device
            gradient_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)

        skip = torch.any(torch.isnan(gradient_norm)) or gradient_norm >= self.config.gradient_skip_threshold
        return bool(skip.cpu().item())

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()

        metrics_accumulator = WeightedMeanMetrics()

        device = get_model_device(self.model)

        epoch_bar = tqdm(self.data_loader, desc="Batch", leave=False, mininterval=0, miniters=1,)

        for batch_num, sample in enumerate(epoch_bar):
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

            skip_epoch = self._skip_gradient_correctness()
            skip_metric = int(skip_epoch)

            if not skip_epoch:
                self.optimizer.step()

                metric_values = dict(batch_loss_metrics.to_dict())
                if "loss" in metric_values:
                    raise ValueError('"loss" is reserved for the optimized loss tensor')
                
                if "epochs_skipped" in metric_values:
                    raise ValueError('"epochs_skipped" is reserved for the optimized loss tensor')

                metric_values["loss"] = loss.detach().item()
                metric_values["epochs_skipped"] = skip_metric
                metrics_accumulator.update(metric_values, batch_size=image.shape[0])
            else:
                metric_values = {}
                metric_values["epochs_skipped"] = skip_metric
                metrics_accumulator.update(metric_values, batch_size=image.shape[0])

            if (batch_num + 1) % self.config.epoch_bar_info_every_batch == 0:
                epoch_bar.set_postfix(metrics_accumulator.compute())

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
            self.param_scheduler.step()

            checkpoint = {
                "epoch": epoch_num,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "param_scheduler_state_dict": self.param_scheduler.state_dict(),
                "metrics_history": metrics_history,
            }
            self.logger.save_last_checkpoint(checkpoint)

            if epoch_num % self.config.log_every_epoch == 0:
                self.logger.save_metrics_plot(
                    metrics_history,
                    epoch_num,
                )

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

