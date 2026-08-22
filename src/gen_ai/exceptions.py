from collections.abc import Callable
from typing import TypeVar


class GenAIError(Exception):
    pass


class ConfigurationError(GenAIError, ValueError):
    pass


class ModelShapeError(GenAIError, ValueError):
    pass


class TrainingError(GenAIError, RuntimeError):
    pass


ErrorT = TypeVar("ErrorT", bound=Exception)


def require(
    condition: bool,
    exception_type: type[ErrorT],
    message: str | Callable[[], str],
) -> None:
    if condition:
        return

    if callable(message):
        message = message()

    raise exception_type(message)
