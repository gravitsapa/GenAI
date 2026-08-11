from typing import Optional

from pathlib import Path
from datetime import datetime

import json

import torch

from gen_ai.util.json_helpers import json_default
from gen_ai.project_config import EXPERIMENTS_DIR

class Logger:
    def __init__(
        self,
        experiment_name: str,
        experiments_dir: Path = EXPERIMENTS_DIR,
    ):
        self.experiment_name = experiment_name
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.experiment_name_full_name = f"{self.experiment_name}_{self.start_time}"
        self.all_experiments_dir = experiments_dir
        
        self.experiment_dir = self.all_experiments_dir / self.experiment_name_full_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.experiment_dir / "checkpoints"

        self.config_name = "config"
        self.metrics_name = "metrics"
        self.checkpoint_name = "checkpoint"


    def log_config(
        self,
        config: dict
    ):
        config_filename = self.experiment_dir / f"{self.config_name}.json"

        with open(config_filename, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=4, default=json_default)


    def log_metrics(
        self,
        metrics: dict
    ):
        metrics_filename = self.experiment_dir / f"{self.metrics_name}.jsonl"

        with open(metrics_filename, "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(metrics, default=json_default, ensure_ascii=False) + "\n")


    def save_checkpoint(
        self,
        checkpoint: dict,
        epoch_num: int,
    ):
        checkpoint_filename = self.checkpoints_dir / f"{self.checkpoint_name}_epoch_{epoch_num:04}.jsonl"

        torch.save(checkpoint, checkpoint_filename)

