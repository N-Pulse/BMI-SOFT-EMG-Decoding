"""Build windowed PyTorch datasets from EMG-EPN-612 participants."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from bmiemg.data.epn612.loader import EMGTrial, load_user_trials
from bmiemg.preprocessing.emg import (
    ChannelNormalizationStats,
    fit_channel_normalization,
    normalize_emg_channels,
)
from bmiemg.preprocessing.segmentation import labelled_interval

from .adaptation import (
    BoardAdaptationConfig,
    adapt_trial,
    board_counts_to_normalized,
)
from .split import MOVEMENTS, select_participant_splits

WindowDataset: TypeAlias = Dataset[tuple[torch.Tensor, torch.Tensor]]
DATASET_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """How labelled intervals are converted into model inputs."""

    size_samples: int = 400
    step_samples: int = 200
    max_per_trial: int | None = 10

    def __post_init__(self) -> None:
        if self.size_samples <= 0 or self.step_samples <= 0:
            raise ValueError("Window size and step must be positive")
        if self.max_per_trial is not None and self.max_per_trial <= 0:
            raise ValueError("max_per_trial must be positive or None")


@dataclass(frozen=True, slots=True)
class EPN612DatasetConfig:
    """All parameters of the EPN-612 input pipeline."""

    root: Path
    participant_id: str | None = None
    max_participants: int | None = None
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    class_labels: tuple[str, ...] = MOVEMENTS
    normalization_epsilon: float = 1e-6
    adaptation: BoardAdaptationConfig = field(default_factory=BoardAdaptationConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)

    def __post_init__(self) -> None:
        if self.participant_id is not None and self.max_participants is not None:
            raise ValueError(
                "participant_id and max_participants are mutually exclusive"
            )
        if self.max_participants is not None and self.max_participants <= 0:
            raise ValueError("max_participants must be a positive integer")
        if len(self.split_ratios) != 3 or any(value < 0 for value in self.split_ratios):
            raise ValueError("split_ratios must contain three non-negative values")
        if not np.isclose(sum(self.split_ratios), 1.0):
            raise ValueError("split_ratios must sum to one")
        if not self.class_labels or len(set(self.class_labels)) != len(
            self.class_labels
        ):
            raise ValueError("class_labels must be non-empty and unique")
        if (
            not np.isfinite(self.normalization_epsilon)
            or self.normalization_epsilon <= 0
        ):
            raise ValueError("normalization_epsilon must be positive")

    @property
    def named_split_ratios(self) -> dict[str, float]:
        """Return ratios keyed by the split module's canonical names."""

        train, validation, test = self.split_ratios
        return {"train": train, "val": validation, "test": test}


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """The three datasets consumed by the training loop."""

    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset
    class_labels: tuple[str, ...]
    participant_count: int
    normalization: ChannelNormalizationStats

    @property
    def sizes(self) -> dict[str, int]:
        """Return sample counts suitable for concise run logging."""

        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def build_epn612_datasets(
    config: EPN612DatasetConfig,
    *,
    random_seed: int,
) -> DatasetSplits:
    """Run participant selection, board adaptation, windowing, and splitting."""

    selection = select_participant_splits(
        config.root,
        participant_id=config.participant_id,
        max_participants=config.max_participants,
        ratios=config.named_split_ratios,
        random_seed=random_seed,
    )
    label_to_index = {label: index for index, label in enumerate(config.class_labels)}
    windows: dict[str, list[np.ndarray]] = {name: [] for name in DATASET_SPLIT_NAMES}
    labels: dict[str, list[int]] = {name: [] for name in DATASET_SPLIT_NAMES}

    if config.participant_id is not None:
        participant = selection.participants[0]
        trials = [
            trial
            for trial in load_user_trials(
                participant.source_path,
                section="trainingSamples",
            )
            if trial.gesture in label_to_index
        ]
        trial_assignments = assign_trial_splits(
            trials,
            ratios=dict(zip(DATASET_SPLIT_NAMES, config.split_ratios, strict=True)),
            random_seed=random_seed,
        )
        for trial in trials:
            split = trial_assignments[trial.trial_id]
            extracted = extract_trial_windows(trial, config)
            windows[split].extend(extracted)
            labels[split].extend([label_to_index[trial.gesture]] * len(extracted))
    else:
        participant_split_names = {
            "train": "train",
            "val": "validation",
            "test": "test",
        }
        for participant in selection.participants:
            split = participant_split_names[
                selection.assignments[participant.participant_id]
            ]
            for trial in load_user_trials(
                participant.source_path,
                section="trainingSamples",
            ):
                if trial.gesture not in label_to_index:
                    continue
                extracted = extract_trial_windows(trial, config)
                windows[split].extend(extracted)
                labels[split].extend([label_to_index[trial.gesture]] * len(extracted))

    normalization = fit_channel_normalization(
        windows["train"],
        epsilon=config.normalization_epsilon,
    )
    datasets = {
        split: _to_tensor_dataset(
            split,
            windows[split],
            labels[split],
            normalization,
        )
        for split in DATASET_SPLIT_NAMES
    }

    result = DatasetSplits(
        datasets["train"],
        datasets["validation"],
        datasets["test"],
        config.class_labels,
        len(selection.participants),
        normalization,
    )
    empty = [name for name, size in result.sizes.items() if size == 0]
    if empty:
        raise ValueError(
            "EPN-612 produced empty dataset splits: "
            f"{', '.join(empty)}. Select more participants or extract more windows."
        )
    return result


def assign_trial_splits(
    trials: Iterable[EMGTrial],
    *,
    ratios: Mapping[str, float],
    random_seed: int,
) -> dict[str, str]:
    """Assign complete trials to gesture-stratified dataset splits."""

    if set(ratios) != set(DATASET_SPLIT_NAMES):
        raise ValueError(f"Ratios must define exactly {DATASET_SPLIT_NAMES}")
    ratio_values = tuple(ratios[name] for name in DATASET_SPLIT_NAMES)
    if any(value < 0 for value in ratio_values) or not np.isclose(
        sum(ratio_values), 1.0
    ):
        raise ValueError("Split ratios must be non-negative and sum to one")

    trials_by_gesture: dict[str, list[str]] = {}
    seen_trial_ids: set[str] = set()
    for trial in trials:
        if trial.gesture is None:
            raise ValueError(f"Trial {trial.trial_id!r} has no gesture label")
        if trial.trial_id in seen_trial_ids:
            raise ValueError(f"Duplicate trial ID {trial.trial_id!r}")
        seen_trial_ids.add(trial.trial_id)
        trials_by_gesture.setdefault(trial.gesture, []).append(trial.trial_id)

    rng = np.random.default_rng(random_seed)
    assignments: dict[str, str] = {}
    for gesture in sorted(trials_by_gesture):
        trial_ids = trials_by_gesture[gesture]
        shuffled = [trial_ids[index] for index in rng.permutation(len(trial_ids))]
        counts = _stratified_split_counts(len(shuffled), ratio_values)
        offset = 0
        for split, count in zip(DATASET_SPLIT_NAMES, counts, strict=True):
            for trial_id in shuffled[offset : offset + count]:
                assignments[trial_id] = split
            offset += count
    return assignments


def _stratified_split_counts(
    total: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]
    base = [0, 0, 0]
    if total >= len(positive):
        for index in positive:
            base[index] = 1

    remaining = total - sum(base)
    weights = [
        max(total * ratio - base[index], 0.0) for index, ratio in enumerate(ratios)
    ]
    if remaining and not any(weights):
        weights = list(ratios)
    weight_total = sum(weights)
    exact = [
        remaining * weight / weight_total if weight_total else 0.0 for weight in weights
    ]
    additional = [int(np.floor(value)) for value in exact]
    remainder = remaining - sum(additional)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(exact[index] - additional[index]), index),
    )
    for index in order[:remainder]:
        additional[index] += 1
    return tuple(base[index] + additional[index] for index in range(3))


def extract_trial_windows(
    trial: EMGTrial,
    config: EPN612DatasetConfig,
) -> list[np.ndarray]:
    """Adapt one trial and return uniformly sampled labelled windows."""

    adapted = adapt_trial(trial, config.adaptation)
    signal = board_counts_to_normalized(
        adapted.emg_counts,
        adapted.adc_resolution_bits,
    )
    source_start, source_stop = labelled_interval(trial)
    scale = adapted.sampling_rate_hz / trial.sampling_rate_hz
    first = round(source_start * scale)
    stop = min(signal.shape[1], round(source_stop * scale))
    starts = window_start_indices(first, stop, config.windows)
    return [
        signal[:, start : start + config.windows.size_samples].copy()
        for start in starts
    ]


def window_start_indices(
    first: int,
    stop: int,
    config: WindowConfig,
) -> np.ndarray:
    """Return valid starts, uniformly subsampled when a trial has too many."""

    starts = np.arange(
        first,
        stop - config.size_samples + 1,
        config.step_samples,
        dtype=np.int64,
    )
    if config.max_per_trial is not None and len(starts) > config.max_per_trial:
        selected = np.linspace(
            0,
            len(starts) - 1,
            config.max_per_trial,
            dtype=int,
        )
        starts = starts[selected]
    return starts


def _to_tensor_dataset(
    split: str,
    windows: list[np.ndarray],
    labels: list[int],
    normalization: ChannelNormalizationStats,
) -> TensorDataset:
    if not windows:
        raise ValueError(f"EPN-612 produced no {split} windows")
    signals = torch.from_numpy(
        normalize_emg_channels(np.stack(windows), normalization)
    ).float()
    targets = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(signals, targets)
