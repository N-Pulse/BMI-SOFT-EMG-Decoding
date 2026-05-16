# ================================================================
# 0. Section: IMPORTS
# ================================================================
from __future__ import annotations

import numpy as np

from typing import Protocol, Self



# ================================================================
# 1. Section: Functions
# ================================================================
class ClassifierEstimator(Protocol):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Self:
        ...
