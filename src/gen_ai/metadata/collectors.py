from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DeclarationMetadata:
    declared_config: dict

    def to_dict(self):
        return self.declared_config


class DeclarationDescribed(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _get_specific_declaration_metadata(self) -> DeclarationMetadata:
        pass

    def get_declaration_metadata_dict(self) -> dict:
        declaration_metadata_dict = self._get_specific_declaration_metadata().to_dict()
        declaration_metadata_dict['class'] = self.__class__.__name__

        return declaration_metadata_dict

    


@dataclass
class RuntimeMetadata:
    effective_config: dict

    def to_dict(self):
        return self.effective_config


class RuntimeDescribed(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_inner_metadata(self) -> RuntimeMetadata:
        pass
