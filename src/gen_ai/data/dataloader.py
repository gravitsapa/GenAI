from dataclasses import dataclass, asdict

from torch.utils.data import DataLoader

from gen_ai.metadata.configuration_collector import ContainingConfiguration
from gen_ai.data.datasets import DescribedImageDataset
from gen_ai.exceptions import ConfigurationError, require


@dataclass(frozen=True, kw_only=True)
class DataloaderConfig:
    batch_size: int
    shuffle: bool
    pin_memory: bool
    num_workers: int
    persistent_workers: bool

    def __post_init__(self) -> None:
        require(
            self.batch_size > 0,
            ConfigurationError,
            f"batch_size must be positive, got {self.batch_size}",
        )
        require(
            self.num_workers >= 0,
            ConfigurationError,
            f"num_workers must be non-negative, got {self.num_workers}",
        )
        require(
            not self.persistent_workers or self.num_workers > 0,
            ConfigurationError,
            "persistent_workers requires num_workers > 0",
        )


class DescribedImageDataLoader(ContainingConfiguration, DataLoader):
    def __init__(self, dataset: DescribedImageDataset, config: DataloaderConfig):
        super().__init__(config=config, dataset=dataset, **asdict(config))


