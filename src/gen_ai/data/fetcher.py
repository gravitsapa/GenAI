from torch.utils.data import DataLoader

from gen_ai.metadata.configuration_collector import ContainingConfiguration

class DataFetcher(ContainingConfiguration):
    def __init__(self, loader: DataLoader):
        super().__init__(config=None)
        self.loader = loader

    def _fetch(self):
        try:
            batch = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            batch = next(self.iter)
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._fetch()

        return batch
