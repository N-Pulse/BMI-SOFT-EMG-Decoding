# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

from typing_extensions import Any

from ..Credentials import Credentials
from ..Request import Request



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DownloadStrategy(ABC):
    @abstractmethod
    def authenticate(self, credentials: Credentials) -> Any:
        pass

    @abstractmethod
    def download(self, request: Request) -> Path:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass
