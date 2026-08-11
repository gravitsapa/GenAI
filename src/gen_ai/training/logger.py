from typing import Optional

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm, trange

from gen_ai.project_config import EXPERIMENTS_DIR
from gen_ai.models.common import get_model_device
from gen_ai.models.generative import ImageGenerativeModel

class Logger:
    def __init__(
        self,
        experiment_name: str,
        experiments_dir: Path = EXPERIMENTS_DIR,
    ):
        self.experiment_name = experiment_name
        self.experiments_dir = experiments_dir
        
        experiment_dir = self.experiments_dir / self.experiment_name
        # if experiment_dir.exists():
        #     raise RuntimeError("Experiment folder already exist")

        experiment_dir.mkdir(parents=True, exist_ok=True)

    def log_config(
        self,
        
    ):
        pass