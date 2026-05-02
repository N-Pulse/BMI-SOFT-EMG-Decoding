# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class EEGStream:
    time_series: np.ndarray
    time_stamps: np.ndarray
    info: dict
    footer: dict
