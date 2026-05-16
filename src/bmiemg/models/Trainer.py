# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from dataclasses import dataclass

from .target_builders import TargetBuilder
from .TrainingResults import TrainingResults
from .ModelRegistry import ModelRegistry
from .Evaluator import Evaluator



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Trainer:
    registry: ModelRegistry
    target_builder: TargetBuilder
    evaluator: Evaluator

    def train (
        self,
        epochs: mne.Epochs,
        features: np.ndarray,
        model_name: str,
        target_name: str
    ) -> TrainingResults:
        y = self.target_builder.build(
            epochs=epochs,
            target_name=target_name,
        )

        estimator = self.registry.create(model_name)

        scores = self.evaluator.cross_validate(
            estimator=estimator,
            x=features,
            y=y,
        )

        estimator.fit(features, y)

        return TrainingResults(
            model_name=model_name,
            target_name=target_name,
            estimator=estimator,
            scores=scores,
            mean_score=float(scores.mean()),
            std_score=float(scores.std()),
        )
