from abc import ABC, abstractmethod

import torch.nn as nn

from gen_ai.exceptions import ConfigurationError, require

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

        require(
            num_groups > 0,
            ValueError,
            f"num_groups must be positive, got {num_groups}",
        )
        require(
            eps > 0,
            ValueError,
            f"eps must be positive, got {eps}",
        )

        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

    def __call__(self, channels: int) -> nn.Module:
        require(
            channels > 0 and channels % self.num_groups == 0,
            ConfigurationError,
            lambda: (
                "num_groups must divide a positive channel count: "
                f"{self.num_groups} does not divide {channels}"
            ),
        )
        return nn.GroupNorm(
            num_groups=self.num_groups,
            num_channels=channels,
            eps=self.eps,
            affine=self.affine
        )

