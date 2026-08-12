from typing import Any, Optional

from dataclasses import is_dataclass, asdict

from gen_ai.exceptions import require


def _is_dataclass_instance(obj):
    return is_dataclass(obj) and not isinstance(obj, type)


class ContainingConfiguration:
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        require(
            self._check_is_config(self.config),
            TypeError,
            f"Config must be a dataclass",
        )


    def _check_is_config(self, config: Any) -> bool:
        return _is_dataclass_instance(config)


    def _get_additional_metadata(self) -> Optional[dict]:
        return


    def _get_attributes_metadata(self):
        metadata = {}

        for name, value in vars(self).items():
            if isinstance(value, ContainingConfiguration):
                metadata[name] = value.get_metadata_dict()

        return metadata


    def get_metadata_dict(self) -> dict:
        metadata_dict = {}
        metadata_dict["class"] = self.__class__.__name__
        metadata_dict["config"] = asdict(self.config)

        attributes_metadata = self._get_attributes_metadata()
        if attributes_metadata:
            metadata_dict["attributes"] = attributes_metadata
        
        additional_metadata_dict = self._get_additional_metadata()
        if additional_metadata_dict is not None:
            metadata_dict['additional_metadata'] = additional_metadata_dict

        return metadata_dict


# @dataclass
# class RuntimeMetadata:
#     effective_config: dict

#     def to_dict(self):
#         return self.effective_config


# class RuntimeDescribed(ABC):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @abstractmethod
#     def get_inner_metadata(self) -> RuntimeMetadata:
#         pass
