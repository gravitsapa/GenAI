from dataclasses import dataclass, asdict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Optimizer

from gen_ai.metadata.collectors import DeclarationDescribed, DeclarationMetadata


@dataclass(frozen=True, kw_only=True)
class CosineAnnealingLRConfig:
    T_max: int
    eta_min: float


class DescribedCosineAnnealingLR(CosineAnnealingLR, DeclarationDescribed):
    def __init__(
        self, 
        optimizer: Optimizer,
        config: CosineAnnealingLRConfig,
    ):
        super().__init__(optimizer=optimizer, **asdict(config))

        self.config = config

    def get_declaration_metadata(self) -> DeclarationMetadata:
        return DeclarationMetadata(asdict(self.config))
