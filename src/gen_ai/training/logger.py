from typing import Optional

from pathlib import Path
from datetime import datetime

import json

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

    def log_config(
        self,
        config: dict
    ):
        config_filename = self.experiment_dir / "config.json"

        with open(config_filename, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=4, default=json_default)
