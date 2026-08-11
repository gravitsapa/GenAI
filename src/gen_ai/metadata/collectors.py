from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DeclarationMetadata:
    declared_config: dict


class DeclarationDescribed:
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_declaration_metadata(self) -> DeclarationMetadata:
        pass


@dataclass
class RuntimeMetadata:
    effective_config: dict


class RuntimeDescribed:
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_runtime_metadata(self) -> RuntimeMetadata:
        pass
