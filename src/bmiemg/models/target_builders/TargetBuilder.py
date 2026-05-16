# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class TargetBuilder(ABC):
    @abstractmethod
    def build(self, epochs: mne.Epochs, target_name: str) -> np.ndarray:
        pass
