# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator

from .model_factories import ModelFactory

from .ClassifierEstimator import ClassifierEstimator



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ModelRegistry:
    _factories: dict[str, ModelFactory] = field(default_factory=dict)

    def register(self, name: str, factory: ModelFactory) -> None:
        if name in self._factories:
            raise ValueError(f"Model '{name}' is already registered.")

        self._factories[name] = factory

    def create(self, name: str) -> ClassifierEstimator:
        if name not in self._factories:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available models: {self.available_models()}"
            )

        return self._factories[name].create()

    def available_models(self) -> list[str]:
        return sorted(self._factories.keys())
