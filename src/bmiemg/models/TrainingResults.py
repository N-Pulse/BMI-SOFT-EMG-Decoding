# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from .ClassifierEstimator import ClassifierEstimator



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class TrainingResults:
    model_name: str
    target_name: str
    estimator: ClassifierEstimator
    scores: np.ndarray
    mean_score: float
    std_score: float
