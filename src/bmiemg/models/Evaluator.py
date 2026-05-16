# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .TrainingConfig import TrainingConfig
from .ClassifierEstimator import ClassifierEstimator



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Evaluator:
    config: TrainingConfig

    def cross_validate(
        self,
        estimator: ClassifierEstimator,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        cv = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
        )

        return cross_val_score(
            estimator,
            x,
            y,
            cv=cv,
            scoring=self.config.scoring,
        )
