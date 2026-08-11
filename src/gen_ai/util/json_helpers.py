from pathlib import Path
from enum import Enum

def json_default(value: object):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    raise TypeError(
        f"{type(value).__name__} is not JSON serializable"
    )
