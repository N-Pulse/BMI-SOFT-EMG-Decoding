"""Factory for a leakage-safe linear SVM pipeline."""

from dataclasses import dataclass

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .ModelFactory import ModelFactory


@dataclass
class SVMFactory(ModelFactory):
    C: float = 1.0
    class_weight: str | dict | None = "balanced"
    max_iter: int = 10_000
    random_state: int = 42

    def create(self) -> Pipeline:
        if self.C <= 0 or self.max_iter <= 0:
            raise ValueError("C and max_iter must be positive")
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                random_state=self.random_state,
            ),
        )
