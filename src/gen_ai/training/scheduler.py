from dataclasses import dataclass, asdict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.optim import Optimizer

from gen_ai.metadata.configuration_collector import ContainingConfiguration

class DescribedScheduler(ContainingConfiguration, LRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass(frozen=True, kw_only=True)
class CosineAnnealingLRConfig:
    T_max: int
    eta_min: float


class DescribedCosineAnnealingLR(DescribedScheduler, CosineAnnealingLR):
    def __init__(
        self, 
        optimizer: Optimizer,
        config: CosineAnnealingLRConfig,
    ):
        super().__init__(config=config, optimizer=optimizer, **asdict(config))

