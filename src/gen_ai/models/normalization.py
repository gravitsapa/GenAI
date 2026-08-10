from abc import ABC, abstractmethod

import torch.nn as nn

class NormalizationFactory(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, channels: int) -> nn.Module:
        pass

class GroupNormalizationFactory(NormalizationFactory):
    def __init__(
        self,
        num_groups: int,
        eps: float=1e-6,
        affine: bool=True,
    ) -> None:
        super().__init__()

        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

    def __call__(self, channels: int) -> nn.Module:
        return nn.GroupNorm(
            num_groups=self.num_groups,
            num_channels=channels,
            eps=self.eps,
            affine=self.affine
        )

