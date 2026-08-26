from pathlib import Path
from enum import Enum

from gen_ai.metadata.configuration_collector import ContainingConfiguration

def json_default(value: object):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, ContainingConfiguration):
        return value.get_metadata_dict()

    raise TypeError(
        f"{type(value).__name__} is not JSON serializable"
    )
