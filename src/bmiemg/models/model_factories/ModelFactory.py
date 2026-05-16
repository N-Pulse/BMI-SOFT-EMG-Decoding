# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..ClassifierEstimator import ClassifierEstimator



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ModelFactory(ABC):
    @abstractmethod
    def create(self) -> ClassifierEstimator:
        pass
