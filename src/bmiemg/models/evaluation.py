"""Multiclass evaluation for EMG gesture predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass(frozen=True, slots=True)
class ClassificationEvaluation:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    labels: tuple[str, ...]
    confusion_matrix: NDArray[np.int_]
    classification_report: dict

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "labels": list(self.labels),
            "classification_report": self.classification_report,
        }


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> ClassificationEvaluation:
    true = np.asarray(y_true, dtype=str)
    predicted = np.asarray(y_pred, dtype=str)
    if true.ndim != 1 or predicted.ndim != 1 or true.shape != predicted.shape:
        raise ValueError("Targets and predictions must be equal one-dimensional arrays")
    if true.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set")
    labels = tuple(str(label) for label in sorted(set(true) | set(predicted)))
    return ClassificationEvaluation(
        accuracy=float(accuracy_score(true, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(true, predicted)),
        macro_f1=float(f1_score(true, predicted, average="macro", zero_division=0)),
        labels=labels,
        confusion_matrix=confusion_matrix(true, predicted, labels=list(labels)),
        classification_report=classification_report(
            true,
            predicted,
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        ),
    )
