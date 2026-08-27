from math import isfinite
from numbers import Integral, Real
from typing import Any, Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass

from gen_ai.exceptions import ConfigurationError, require
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

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch_now": self._epoch_now,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self._epoch_now = state_dict["epoch_now"]

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

    def __post_init__(self) -> None:
        require(
            isinstance(self.begin_step, Integral) and not isinstance(self.begin_step, bool),
            ConfigurationError,
            f"begin_step must be an integer, got {self.begin_step!r}",
        )
        require(
            isinstance(self.end_step, Integral) and not isinstance(self.end_step, bool),
            ConfigurationError,
            f"end_step must be an integer, got {self.end_step!r}",
        )
        require(
            self.begin_step >= 0,
            ConfigurationError,
            f"begin_step must be >= 0, got {self.begin_step}",
        )
        require(
            self.end_step > self.begin_step,
            ConfigurationError,
            lambda: (
                "end_step must be greater than begin_step, got "
                f"begin_step={self.begin_step}, end_step={self.end_step}"
            ),
        )
        require(
            isinstance(self.begin_value, Real) and isfinite(self.begin_value),
            ConfigurationError,
            f"begin_value must be a finite real number, got {self.begin_value!r}",
        )
        require(
            isinstance(self.end_value, Real) and isfinite(self.end_value),
            ConfigurationError,
            f"end_value must be a finite real number, got {self.end_value!r}",
        )


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
