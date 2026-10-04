"""Command-line training for the EMG-EPN-612 linear SVM."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from bmiemg.data.epn612 import EMGTrial, iter_epn612_trials
from bmiemg.models.evaluation import ClassificationEvaluation, evaluate_classification
from bmiemg.models.model_factories import SVMFactory
from bmiemg.pipeline import EMGSVMDecoder
from bmiemg.preprocessing import (
    EMGProcessingConfig,
    extract_emg_features,
    iter_trial_windows,
    milliseconds_to_samples,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return dict(value)


def collect_feature_dataset(
    trials: Iterable[EMGTrial],
    processing: EMGProcessingConfig,
    step_size_samples: int,
    max_windows_per_trial: int | None,
    allowed_users: set[str] | None = None,
    max_users: int | None = None,
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """Extract compact features while loading one user JSON at a time."""

    feature_batches: list[np.ndarray] = []
    labels: list[str] = []
    users: set[str] = set()
    for trial in trials:
        if allowed_users is not None and trial.user_id not in allowed_users:
            continue
        if allowed_users is None and trial.user_id not in users:
            if max_users is not None and len(users) >= max_users:
                break
            users.add(trial.user_id)
        elif allowed_users is not None:
            users.add(trial.user_id)

        if not np.isclose(trial.sampling_rate_hz, processing.sampling_rate_hz):
            raise ValueError(
                f"{trial.user_id}:{trial.trial_id} is sampled at "
                f"{trial.sampling_rate_hz} Hz, expected {processing.sampling_rate_hz} Hz"
            )
        trial_windows = list(iter_trial_windows(
            trial,
            processing.window_size_samples,
            step_size_samples,
            max_windows_per_trial,
        ))
        if trial_windows:
            signals = np.stack([window.signal for window in trial_windows])
            feature_batches.append(extract_emg_features(signals, processing))
            labels.extend(window.gesture for window in trial_windows)
    if not feature_batches:
        raise ValueError("No labelled windows were produced")
    return np.concatenate(feature_batches), np.asarray(labels, dtype=str), users


def train_from_config(config: Mapping[str, Any]) -> ClassificationEvaluation:
    """Train, evaluate and save one subject-dependent development model."""

    data = _mapping(config, "data")
    processing_values = _mapping(config, "processing")
    filter_values = _mapping(processing_values, "software_filter")
    model_values = _mapping(config, "model")
    output = _mapping(config, "output")

    sampling_rate = float(processing_values["expected_sampling_rate_hz"])
    window_size = milliseconds_to_samples(
        float(processing_values["window_size_ms"]), sampling_rate
    )
    step_size = milliseconds_to_samples(
        float(processing_values["window_step_ms"]), sampling_rate
    )
    high_value = filter_values.get("high_hz")
    processing = EMGProcessingConfig(
        sampling_rate_hz=sampling_rate,
        window_size_samples=window_size,
        feature_names=tuple(str(value) for value in processing_values["features"]),
        software_filter_enabled=bool(filter_values["enabled"]),
        filter_low_hz=float(filter_values["low_hz"]),
        filter_high_hz=None if high_value is None else float(high_value),
        filter_order=int(filter_values["order"]),
    )
    max_users_value = data.get("max_users")
    max_users = None if max_users_value is None else int(max_users_value)
    max_windows_value = processing_values.get("max_windows_per_trial")
    max_windows = None if max_windows_value is None else int(max_windows_value)
    root = Path(str(data["root"]))
    group = cast(Any, str(data["json_group"]))

    training_trials = iter_epn612_trials(
        root, group, cast(Any, str(data["train_section"]))
    )
    train_features, train_labels, selected_users = collect_feature_dataset(
        training_trials, processing, step_size, max_windows, max_users=max_users
    )
    testing_trials = iter_epn612_trials(
        root, group, cast(Any, str(data["test_section"]))
    )
    test_features, test_labels, _ = collect_feature_dataset(
        testing_trials,
        processing,
        step_size,
        max_windows,
        allowed_users=selected_users,
    )

    decoder = EMGSVMDecoder(
        processing,
        SVMFactory(
            C=float(model_values["C"]),
            class_weight=model_values.get("class_weight"),
            max_iter=int(model_values["max_iter"]),
            random_state=int(model_values["random_state"]),
        ),
    ).fit_features(train_features, train_labels)
    evaluation = evaluate_classification(
        test_labels, decoder.predict_features(test_features)
    )

    output_directory = Path(str(output["directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    decoder.save(output_directory / str(output["model_filename"]))
    _save_results(
        evaluation,
        output_directory / str(output["metrics_filename"]),
        output_directory / str(output["confusion_matrix_filename"]),
        train_windows=len(train_features),
        test_windows=len(test_features),
        users=len(selected_users),
    )
    return evaluation


def _save_results(
    evaluation: ClassificationEvaluation,
    metrics_path: Path,
    matrix_path: Path,
    train_windows: int,
    test_windows: int,
    users: int,
) -> None:
    metrics = evaluation.as_dict()
    metrics.update(
        train_window_count=train_windows,
        test_window_count=test_windows,
        user_count=users,
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with matrix_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *evaluation.labels])
        for label, row in zip(
            evaluation.labels, evaluation.confusion_matrix.tolist(), strict=True
        ):
            writer.writerow([label, *row])


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing configuration mapping {key!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EMG-EPN-612 SVM")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/epn612_svm.yml")
    )
    arguments = parser.parse_args()
    result = train_from_config(load_config(arguments.config))
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
