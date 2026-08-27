from collections.abc import Callable, Mapping
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
    resume_from_checkpoint: Path | None = None

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

        if self.resume_from_checkpoint is not None:
            checkpoint_path = Path(self.resume_from_checkpoint)
            require(
                checkpoint_path.is_file(),
                ConfigurationError,
                f"resume_from_checkpoint must point to an existing checkpoint file, got {checkpoint_path}",
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

    def _load_checkpoint(
        self,
        device: torch.device,
    ) -> tuple[int, list[dict], list[int]]:
        if self.config.resume_from_checkpoint is None:
            return 0, [], []

        checkpoint_path = Path(self.config.resume_from_checkpoint)
        try:
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=False,
                )
            except TypeError:
                # weights_only was introduced after older supported PyTorch releases.
                checkpoint = torch.load(checkpoint_path, map_location=device)
        except (OSError, RuntimeError, ValueError) as error:
            raise TrainingError(
                f"could not load checkpoint from {checkpoint_path}: {error}"
            ) from error

        require(
            isinstance(checkpoint, Mapping),
            TrainingError,
            f"checkpoint at {checkpoint_path} must be a mapping",
        )

        required_keys = {
            "step",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "param_scheduler_state_dict",
        }
        missing_keys = required_keys - checkpoint.keys()
        require(
            not missing_keys,
            TrainingError,
            lambda: (
                f"checkpoint at {checkpoint_path} is missing required keys: "
                f"{', '.join(sorted(missing_keys))}"
            ),
        )

        checkpoint_step = checkpoint["step"]
        require(
            isinstance(checkpoint_step, int) and not isinstance(checkpoint_step, bool),
            TrainingError,
            f"checkpoint step must be an integer, got {checkpoint_step!r}",
        )
        require(
            0 <= checkpoint_step < self.config.num_steps,
            TrainingError,
            lambda: (
                f"checkpoint step {checkpoint_step} must be in [0, {self.config.num_steps}) "
                "to continue training"
            ),
        )

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.param_scheduler.load_state_dict(checkpoint["param_scheduler_state_dict"])
        except (RuntimeError, ValueError, KeyError) as error:
            raise TrainingError(
                f"checkpoint at {checkpoint_path} is incompatible with the current training setup: {error}"
            ) from error

        metrics_history = checkpoint.get("metrics_history", [])
        steps_history = checkpoint.get("steps_history", [])
        if not isinstance(metrics_history, list) or not isinstance(steps_history, list):
            raise TrainingError(
                f"checkpoint at {checkpoint_path} contains invalid metrics history"
            )
        if len(metrics_history) != len(steps_history):
            metrics_history, steps_history = [], []

        print(f"Resumed training from {checkpoint_path} at step {checkpoint_step}")
        return checkpoint_step, metrics_history, steps_history

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
        checkpoint_step, metrics_history, steps_history = self._load_checkpoint(device)
        self.model.train()

        metrics_accumulator = WeightedMeanMetrics()

        steps_bar = trange(
            checkpoint_step + 1,
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
                    "steps_history": steps_history,
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
