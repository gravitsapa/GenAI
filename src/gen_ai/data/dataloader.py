from dataclasses import dataclass, asdict

from torch.utils.data import DataLoader

from gen_ai.metadata.collectors import DeclarationDescribed, DeclarationMetadata
from gen_ai.data.datasets import DescribedImageDataset


@dataclass(frozen=True, kw_only=True)
class DataloaderConfig:
    batch_size: int
    shuffle: bool
    pin_memory: bool
    num_workers: int


class DescribedImageDataLoader(DataLoader, DeclarationDescribed):
    def __init__(self, dataset: DescribedImageDataset, config: DataloaderConfig):
        super().__init__(dataset=dataset, **asdict(config))

        self.config = config

    def get_declaration_metadata(self) -> DeclarationMetadata:
        return DeclarationMetadata(asdict(self.config))

