from dataclasses import dataclass, asdict

import torch
from torch.optim import Adam
from torch.optim.optimizer import ParamsT

from gen_ai.metadata.collectors import DeclarationDescribed, DeclarationMetadata


@dataclass(frozen=True, kw_only=True)
class AdamConfig:
    lr: float


class DescribedAdam(Adam, DeclarationDescribed):
    def __init__(
        self, 
        params: ParamsT,
        config: AdamConfig,
    ):
        super().__init__(params=params, **asdict(config))

        self.config = config

    def get_declaration_metadata(self) -> DeclarationMetadata:
        return DeclarationMetadata(asdict(self.config))
