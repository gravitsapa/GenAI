from dataclasses import dataclass, asdict
from typing import Optional

import torch
from torchvision.transforms import v2

from gen_ai.data.image import ImageShape
from gen_ai.metadata.configuration_collector import ContainingConfiguration


@dataclass(frozen=True, kw_only=True)
class AugmentationsConfig:
    random_crop_scale: Optional[float] = 1.2
    random_horizontal_flip: Optional[float] = 0.5


class AugmentationBuilder(ContainingConfiguration):
    def __init__(
        self,
        config: AugmentationsConfig,
    ):
        super().__init__(config=config)

    def build(self, image_shape: ImageShape) -> v2.Transform:
        augmentation_list = []
        if self.config.random_crop_scale is not None:
            scaled_size = tuple(int(size * self.config.random_crop_scale) for size in image_shape)
            augmentation_list.append(v2.Resize(scaled_size))
            augmentation_list.append(v2.RandomCrop(image_shape))
        else:
            augmentation_list.append(v2.Resize(image_shape))
        if self.config.random_horizontal_flip is not None:
            augmentation_list.append(v2.RandomHorizontalFlip(p=self.config.random_horizontal_flip))
        return v2.Compose(augmentation_list)

