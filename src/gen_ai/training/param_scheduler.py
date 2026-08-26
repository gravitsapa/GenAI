from typing import Any, Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass

from gen_ai.metadata.configuration_collector import ContainingConfiguration

class ParamScheduler:
    def __init__(
        self,
        epoch_now: int = 1,
    ):
        self._epoch_now = epoch_now

    def step(self):
        self._epoch_now += 1

    def get_epoch(self) -> int:
        return self._epoch_now

T = TypeVar('T')

class Scheme(ABC, Generic[T]):
    @abstractmethod
    def __call__(
        self,
        param_scheduler: ParamScheduler,
    ) -> T:
        pass


NumericT = TypeVar('NumericT', int, float)
def clamp(value: NumericT, lower_bound: NumericT, upper_bound: NumericT) -> NumericT:
    return max(lower_bound, min(value, upper_bound))


@dataclass(frozen=True, kw_only=True)
class LinearFloatSchemeConfig:
    begin_step: int
    end_step: int
    begin_value: float
    end_value: float


class LinearFloatScheme(ContainingConfiguration, Scheme[float]):
    def __init__(
        self,
        config: LinearFloatSchemeConfig,
    ):
        super().__init__(config=config)

    def __call__(
        self,
        param_scheduler: ParamScheduler,
    ) -> float:
        epoch_now = param_scheduler.get_epoch()

        alpha = clamp(
            float(epoch_now - self.config.begin_step) / (self.config.end_step - self.config.begin_step),
            0.,
            1,
        )

        return alpha * self.config.end_value + (1. - alpha) * self.config.begin_value


U = TypeVar('U')
class ScheduledParam(ContainingConfiguration, Generic[U]):
    def __init__(
        self,
        param_scheduler: ParamScheduler,
        scheme: Scheme[U],
    ):
        super().__init__(config=None)

        self._param_scheduler = param_scheduler
        self._scheme = scheme

    def __call__(self) -> U:
        return self._scheme(self._param_scheduler)

V = TypeVar('V')
def get_param(param: V | ScheduledParam[V]) -> V:
    if isinstance(param, ScheduledParam):
        return param()
    return param
