from collections import defaultdict
from collections.abc import Mapping
from typing import Protocol


class LossMetrics(Protocol):
    def to_dict(self) -> Mapping[str, float]: ...


class WeightedMeanMetrics:
    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._weights: dict[str, int] = defaultdict(int)

    def update(self, metrics: Mapping[str, float], batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for name, value in metrics.items():
            self._sums[name] += float(value) * batch_size
            self._weights[name] += batch_size

    def compute(self) -> dict[str, float]:
        if not self._sums:
            raise ValueError("cannot compute metrics without batches")

        return {
            name: self._sums[name] / self._weights[name]
            for name in self._sums
        }

    def clear(self) -> None:
        self._sums = defaultdict(float)
        self._weights = defaultdict(int)
