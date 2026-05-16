# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .ModelFactory import ModelFactory
from ..ClassifierEstimator import ClassifierEstimator



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class LogisticRegressionFactory(ModelFactory):
    random_state: int
    class_weight: str
    max_iter: int

    def create(self) -> ClassifierEstimator:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
            ),
        )
