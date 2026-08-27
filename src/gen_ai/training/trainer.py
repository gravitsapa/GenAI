from collections.abc import Callable
from typing import Any, Optional
from dataclasses import dataclass, asdict

from pathlib import Path

import torch
from tqdm.auto import tqdm, trange

from gen_ai.metadata.configuration_collector import ContainingConfiguration

from gen_ai.data.fetcher import DataFetcher
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
    num_steps: int
    eval_and_save_every_step: int
    gradient_skip_threshold: float
    collect_metrics_every_step: int

    def __post_init__(self) -> None:
        require(
            self.num_steps > 0,
            ConfigurationError,
            f"num_steps must be positive, got {self.num_steps}",
        )
        require(
            self.eval_and_save_every_step > 0,
            ConfigurationError,
            f"eval_and_save_every_step must be positive, got {self.eval_and_save_every_step}",
        )
        require(
            self.collect_metrics_every_step > 0,
            ConfigurationError,
            f"collect_metrics_every_step must be positive, got {self.collect_metrics_every_step}",
        )


class Trainer(ContainingConfiguration):
    def __init__(
        self,
        model: DescribedImageGenerativeModel,
        data_fetcher: DataFetcher,
        loss_function: Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]],
        optimizer: DescribedOptimizer,
        logger: Logger,
        scheduler: DescribedScheduler,
        param_scheduler: ParamScheduler,
        config: TrainerConfig,
    ):
        super().__init__(config=config)

        self.model = model
        self.data_fetcher = data_fetcher
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

    def _train_one_step(
        self, 
        metrics_accumulator: WeightedMeanMetrics,
        device: torch.device,
    ) -> None:
        sample = next(self.data_fetcher)

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

        self.scheduler.step()
        self.param_scheduler.step()

    def _collect_metrics(
        self,
        step_num: int,
        loss_metrics: dict[str, float],
    ) -> dict:
        return {
            "step": step_num,
            "learning_rates": [
                parameter_group["lr"]
                for parameter_group in self.optimizer.param_groups
            ],
            **loss_metrics,
        }


    def train_loop(
        self,
    ) -> list[dict]:
        self.logger.log_config(self.get_metadata_dict())

        sampler = Sampler(self.model)
        device = get_model_device(self.model)
        self.model.train()

        metrics_accumulator = WeightedMeanMetrics()
        metrics_history: list[dict] = []
        steps_history: list[int] = []

        steps_bar = trange(
            1,
            self.config.num_steps + 1,
            desc="Step",
            leave=True,
            mininterval=1,
            miniters=self.config.collect_metrics_every_step,
        )

        for step_num in steps_bar:
            self._train_one_step(
                metrics_accumulator,
                device,
            )
            
            if step_num % self.config.collect_metrics_every_step == 0:
                metrics = metrics_accumulator.compute()
                metrics_accumulator.clear()

                steps_bar.set_postfix(metrics)
                metrics_history.append(metrics)
                steps_history.append(step_num)

                self.logger.log_metrics(
                    self._collect_metrics(step_num, metrics)
                )

            if step_num % self.config.eval_and_save_every_step == 0:
                checkpoint = {
                    "step": step_num,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "param_scheduler_state_dict": self.param_scheduler.state_dict(),
                    "metrics_history": metrics_history,
                }
                self.logger.save_last_checkpoint(checkpoint)

                self.logger.save_metrics_plot(
                    steps_history,
                    metrics_history,
                    step_num,
                )

                self.logger.save_checkpoint(
                    checkpoint,
                    step_num,
                )

                samples_fig = sampler.sample_grid()
                self.model.train()
                self.logger.save_samples(
                    samples_fig,
                    step_num,
                )

        return metrics_history
