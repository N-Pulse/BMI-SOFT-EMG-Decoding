"""Serializable shared training and online-inference pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from bmiemg.models.model_factories import SVMFactory
from bmiemg.preprocessing import (
    EMGProcessingConfig,
    expanded_feature_names,
    extract_emg_features,
)


@dataclass(slots=True)
class EMGSVMDecoder:
    """Feature extraction, fitted scaler and fitted SVM in one artifact."""

    processing: EMGProcessingConfig
    svm_factory: SVMFactory = field(default_factory=SVMFactory)
    estimator: Pipeline | None = field(default=None, init=False)
    classes_: tuple[str, ...] = field(default=(), init=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return expanded_feature_names(self.processing)

    def fit(self, windows: np.ndarray, labels: np.ndarray) -> "EMGSVMDecoder":
        features = extract_emg_features(windows, self.processing)
        return self.fit_features(features, labels)

    def fit_features(
        self, features: np.ndarray, labels: np.ndarray
    ) -> "EMGSVMDecoder":
        """Fit precomputed features, used by memory-efficient offline training."""

        feature_array = self._validate_features(features)
        targets = np.asarray(labels, dtype=str)
        if targets.ndim != 1 or len(targets) != len(feature_array):
            raise ValueError("There must be one label per EMG window")
        if len(np.unique(targets)) < 2:
            raise ValueError("SVM training requires at least two classes")
        self.estimator = self.svm_factory.create()
        self.estimator.fit(feature_array, targets)
        self.classes_ = tuple(str(value) for value in self.estimator.classes_)
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        features = extract_emg_features(windows, self.processing)
        return self.predict_features(features)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        feature_array = self._validate_features(features)
        return np.asarray(self._fitted().predict(feature_array), dtype=str)

    def predict_one(self, window: np.ndarray) -> str:
        predictions = self.predict(window)
        if predictions.shape != (1,):
            raise RuntimeError("predict_one received more than one window")
        return str(predictions[0])

    def save(self, path: str | Path) -> Path:
        self._fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "EMGSVMDecoder":
        value = joblib.load(Path(path))
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        value._fitted()
        return value

    def _fitted(self) -> Pipeline:
        if self.estimator is None:
            raise RuntimeError("Decoder is not fitted")
        return self.estimator

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        expected_columns = len(self.feature_names)
        if array.ndim != 2 or array.shape[1] != expected_columns or len(array) == 0:
            raise ValueError(
                f"Expected feature matrix (n_windows, {expected_columns}), got "
                f"{array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Features contain non-finite values")
        return array


@dataclass(slots=True)
class OnlineEMGInference:
    """Rolling live buffer that emits predictions at a fixed step."""

    decoder: EMGSVMDecoder
    step_size_samples: int
    _buffer: deque[np.ndarray] = field(init=False)
    _countdown: int = field(init=False)

    def __post_init__(self) -> None:
        if self.step_size_samples <= 0:
            raise ValueError("step_size_samples must be positive")
        self.reset()

    def reset(self) -> None:
        self._buffer = deque(maxlen=self.decoder.processing.window_size_samples)
        self._countdown = self.decoder.processing.window_size_samples

    def add_samples(self, samples: np.ndarray) -> list[str]:
        """Consume ``(8, n_samples)`` data and return zero or more labels."""

        array = np.asarray(samples, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, np.newaxis]
        if array.ndim != 2 or array.shape[0] != 8:
            raise ValueError(f"Expected online samples shaped (8, n), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Online samples contain non-finite values")

        predictions: list[str] = []
        for column in array.T:
            self._buffer.append(column.copy())
            self._countdown -= 1
            if len(self._buffer) == self._buffer.maxlen and self._countdown == 0:
                predictions.append(
                    self.decoder.predict_one(np.stack(tuple(self._buffer), axis=1))
                )
                self._countdown = self.step_size_samples
        return predictions
