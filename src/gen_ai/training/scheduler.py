from dataclasses import dataclass, asdict
from typing import Sequence

import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LRScheduler, SequentialLR, LinearLR
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


@dataclass(frozen=True, kw_only=True)
class LinearLRConfig:
    total_iters: int
    start_factor: float = 0.01
    end_factor: float = 1.0


class DescribedLinearLR(DescribedScheduler, LinearLR):
    def __init__(
        self, 
        optimizer: Optimizer,
        config: LinearLRConfig,
    ):
        super().__init__(config=config, optimizer=optimizer, **asdict(config))


@dataclass(frozen=True, kw_only=True)
class ConstantLRConfig:
    total_iters: int
    factor: float = 1.


class DescribedConstantLR(DescribedScheduler, ConstantLR):
    def __init__(
        self,
        optimizer: Optimizer,
        config: ConstantLRConfig,
    ):
        super().__init__(config=config, optimizer=optimizer, **asdict(config))


class DescribedSequentialLR(DescribedScheduler, SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: Sequence[DescribedScheduler],
        milestones: Sequence[int],
    ):
        self.described_schedulers = tuple(schedulers)
        self.described_milestones = tuple(milestones)
        
        super().__init__(
            config=None, 
            optimizer=optimizer,
            schedulers=schedulers,
            milestones=milestones
        )

    def _get_additional_metadata(self) -> dict | None:
        return {
            'schedulers': [
                scheduler.get_metadata_dict()
                for scheduler in self.described_schedulers
            ],
            'milestones': self.described_milestones
        }

