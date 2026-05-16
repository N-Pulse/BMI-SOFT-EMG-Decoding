# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier

from .ModelFactory import ModelFactory



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DecisionTreeFactory(ModelFactory):
    random_state: int
    max_depth: int | None = None
    class_weight: str | None = "balanced"

    def create(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            random_state=self.random_state,
            class_weight=self.class_weight,
            max_depth=self.max_depth
        )
