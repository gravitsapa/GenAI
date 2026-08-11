from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod

from PIL.Image import Image

import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import v2

from datasets import load_dataset

from gen_ai.metadata.collectors import DeclarationDescribed, DeclarationMetadata

from gen_ai.project_config import DATA_DIR
from gen_ai.data.image_sample import ImageSample


class ImageDataset(ABC, Dataset):
    def __init__(self, image_shape: tuple[int, int]):
        self.image_shape = image_shape

        self.cut_transform = v2.Resize(image_shape)
        self.pil_to_tensor_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> ImageSample:
        pass

    def tensor_from_pil(self, pil_image: Image) -> Tensor:
        transform = v2.Compose([
            self.cut_transform,
            self.pil_to_tensor_transform,
        ])

        return transform(pil_image)


class DescribedImageDataset(ImageDataset, DeclarationDescribed):
    pass


@dataclass(frozen=True, kw_only=True)
class ImageDatasetConfig:
    image_shape: tuple[int, int]
    data_dir: Path=DATA_DIR


class AnimeFaces256(DescribedImageDataset):
    def __init__(self, image_dataset_config: ImageDatasetConfig):
        super().__init__(image_dataset_config.image_shape)

        self.config = image_dataset_config

        self.dataset_dir = self.config.data_dir / "anime_faces_256"

        self.data = load_dataset(
            "puruchinera/anime-faces-256",
            cache_dir=str(self.dataset_dir),
        )


    def __len__(self):
        return len(self.data['train'])


    def __getitem__(self, index):
        pil_image = self.data['train'][index]['image']
        return ImageSample(
            image=self.tensor_from_pil(pil_image),
            tags=self.data['train'][index]['tags'],
        )

    
    def _get_specific_declaration_metadata(self) -> DeclarationMetadata:
        return DeclarationMetadata(asdict(self.config))
