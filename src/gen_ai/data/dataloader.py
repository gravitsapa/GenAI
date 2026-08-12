from dataclasses import dataclass, asdict

from torch.utils.data import DataLoader

from gen_ai.metadata.configuration_collector import ContainingConfiguration
from gen_ai.data.datasets import DescribedImageDataset


@dataclass(frozen=True, kw_only=True)
class DataloaderConfig:
    batch_size: int
    shuffle: bool
    pin_memory: bool
    num_workers: int
    persistent_workers: bool


class DescribedImageDataLoader(ContainingConfiguration, DataLoader):
    def __init__(self, dataset: DescribedImageDataset, config: DataloaderConfig):
        super().__init__(config=config, dataset=dataset, **asdict(config))


