from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from gen_ai.config import DATA_DIR
from pathlib import Path
from torch import Tensor
from abc import ABC, abstractmethod
from PIL.Image import Image
from torchvision.transforms import v2
from gen_ai.data.image_sample import ImageSample


class ImageDataset(ABC, Dataset):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

        self.cut_transform = v2.Resize((self.height, self.width))
        self.pil_to_tensor_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
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


class AnimeFaces256(ImageDataset):
    def __init__(self, height: int, width: int, data_dir: Path=DATA_DIR):
        super().__init__(height, width)

        self.dataset_dir = DATA_DIR / "anime_faces_256"

        self.data = load_dataset(
            "puruchinera/anime-faces-256",
            cache_dir=str(self.dataset_dir),
        )

    def __len__(self):
        return len(self.data['train'])

    def __getitem__(self, index):
        pil_image = self.data['train'][index]['image']
        return ImageSample(
            image = self.tensor_from_pil(pil_image),
            tags = self.data['train'][index]['tags'],
        )

