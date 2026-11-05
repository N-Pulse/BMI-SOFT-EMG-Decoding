# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .download_spec import DownloadSpec


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class DownloadStrategy[SpecT: DownloadSpec](ABC):

    @abstractmethod
    def validate(self, spec: SpecT) -> None:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, spec: SpecT) -> None:
        raise NotImplementedError
