from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.optim.optimizer import ParamsT

from gen_ai.metadata.configuration_collector import ContainingConfiguration


class DescribedOptimizer(ContainingConfiguration, Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass(frozen=True, kw_only=True)
class AdamConfig:
    lr: float


class DescribedAdam(DescribedOptimizer, Adam):
    def __init__(
        self, 
        params: ParamsT,
        config: AdamConfig,
    ):
        super().__init__(config=config, params=params, **asdict(config))

