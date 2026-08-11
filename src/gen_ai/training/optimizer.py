from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.optim.optimizer import ParamsT

from gen_ai.metadata.collectors import DeclarationDescribed, DeclarationMetadata


class DescribedOptimizer(Optimizer, DeclarationDescribed):
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
        super().__init__(params=params, **asdict(config))

        self.config = config

    def _get_specific_declaration_metadata(self) -> DeclarationMetadata:
        return DeclarationMetadata(asdict(self.config))
