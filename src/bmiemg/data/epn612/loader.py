"""Lazy, validated loading of EMG-EPN-612 participant JSON files."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int_]
DatasetSection: TypeAlias = Literal["trainingSamples", "testingSamples"]
JsonGroup: TypeAlias = Literal["trainingJSON", "testingJSON"]

EMG_CHANNEL_NAMES = tuple(f"ch{index}" for index in range(1, 9))
DATASET_SECTIONS = ("trainingSamples", "testingSamples")
JSON_GROUPS = ("trainingJSON", "testingJSON")


@dataclass(frozen=True, slots=True)
class EMGTrial:
    """One variable-length recording and its offline-only annotations."""

    emg: FloatArray  # (8, n_samples); the only source of model inputs
    sampling_rate_hz: float
    user_id: str
    trial_id: str
    dataset_section: DatasetSection
    gesture: str | None
    ground_truth: IntArray | None
    ground_truth_index: tuple[int, int] | None
    gesture_start: int | None
    source_path: Path

    def __post_init__(self) -> None:
        if self.emg.ndim != 2 or self.emg.shape[0] != 8 or self.emg.shape[1] == 0:
            raise ValueError(
                f"EMG must have shape (8, n_samples), got {self.emg.shape} "
                f"at {self.source_path}:{self.trial_id}"
            )
        if not np.isfinite(self.emg).all():
            raise ValueError(f"Non-finite EMG at {self.source_path}:{self.trial_id}")
        if self.sampling_rate_hz <= 0:
            raise ValueError("Sampling rate must be positive")


def load_user_trials(
    json_path: str | Path,
    section: DatasetSection = "trainingSamples",
) -> Iterator[EMGTrial]:
    """Load one participant document and yield its trials in numeric order."""

    path = Path(json_path)
    _validate_choice("section", section, DATASET_SECTIONS)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")

    general = _mapping(document, "generalInfo", path)
    user = _mapping(document, "userInfo", path)
    samples = _mapping(document, section, path)
    sampling_rate = _positive_float(
        general.get("samplingFrequencyInHertz"), "samplingFrequencyInHertz", path
    )
    user_id = _string(user.get("name"), "userInfo.name", path)

    for trial_id, value in sorted(samples.items(), key=lambda item: _numeric_key(item[0])):
        if not isinstance(trial_id, str) or not isinstance(value, Mapping):
            raise ValueError(f"Invalid trial entry {trial_id!r} in {path}")
        sample = cast(Mapping[str, Any], value)
        yield _parse_trial(sample, sampling_rate, user_id, trial_id, section, path)


def iter_epn612_trials(
    dataset_root: str | Path,
    json_group: JsonGroup = "trainingJSON",
    section: DatasetSection = "trainingSamples",
) -> Iterator[EMGTrial]:
    """Yield trials while retaining no more than one user JSON in memory."""

    root = Path(dataset_root)
    _validate_choice("json_group", json_group, JSON_GROUPS)
    _validate_choice("section", section, DATASET_SECTIONS)
    group_path = root / json_group
    if not group_path.is_dir():
        raise FileNotFoundError(group_path)

    paths = sorted(group_path.glob("user*/*.json"), key=_user_key)
    if not paths:
        raise FileNotFoundError(f"No user JSON files under {group_path}")
    for path in paths:
        yield from load_user_trials(path, section)


def _parse_trial(
    sample: Mapping[str, Any],
    sampling_rate: float,
    user_id: str,
    trial_id: str,
    section: DatasetSection,
    path: Path,
) -> EMGTrial:
    emg_object = _mapping(sample, "emg", path, trial_id)
    channels: list[FloatArray] = []
    for name in EMG_CHANNEL_NAMES:
        try:
            channel = np.asarray(emg_object[name], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing or invalid {name} at {path}:{trial_id}") from error
        if channel.ndim != 1:
            raise ValueError(f"{name} is not one-dimensional at {path}:{trial_id}")
        channels.append(channel)
    lengths = {len(channel) for channel in channels}
    if len(lengths) != 1:
        raise ValueError(f"Unequal EMG channel lengths at {path}:{trial_id}: {lengths}")

    gesture_value = sample.get("gestureName")
    gesture = None if gesture_value is None else _string(
        gesture_value, "gestureName", path, trial_id
    )
    return EMGTrial(
        emg=np.stack(channels).astype(np.float32, copy=False),
        sampling_rate_hz=sampling_rate,
        user_id=user_id,
        trial_id=trial_id,
        dataset_section=section,
        gesture=gesture,
        ground_truth=_optional_int_array(sample.get("groundTruth"), path, trial_id),
        ground_truth_index=_optional_interval(
            sample.get("groundTruthIndex"), path, trial_id
        ),
        gesture_start=_optional_int(
            sample.get("startPointforGestureExecution"), path, trial_id
        ),
        source_path=path,
    )


def _mapping(
    parent: Mapping[str, Any], key: str, path: Path, trial_id: str | None = None
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing object {key!r} at {path}:{trial_id or ''}")
    return cast(Mapping[str, Any], value)


def _string(value: Any, field: str, path: Path, trial_id: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid {field} at {path}:{trial_id or ''}")
    return value


def _positive_float(value: Any, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} in {path}") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"Invalid {field} in {path}: {value!r}")
    return result


def _optional_int_array(value: Any, path: Path, trial_id: str) -> IntArray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.int_)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid groundTruth at {path}:{trial_id}") from error
    if result.ndim != 1:
        raise ValueError(f"groundTruth must be one-dimensional at {path}:{trial_id}")
    return result


def _optional_interval(
    value: Any, path: Path, trial_id: str
) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Invalid groundTruthIndex at {path}:{trial_id}")
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid groundTruthIndex at {path}:{trial_id}") from error
    if start < 0 or end < start:
        raise ValueError(f"Invalid groundTruthIndex at {path}:{trial_id}")
    return start, end


def _optional_int(value: Any, path: Path, trial_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid gesture start at {path}:{trial_id}")
    return value


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"Invalid {name} {value!r}; expected one of {choices}")


def _numeric_key(value: Any) -> tuple[int, str]:
    text = str(value)
    suffix = text.rpartition("_")[2]
    return (int(suffix), text) if suffix.isdigit() else (2**63 - 1, text)


def _user_key(path: Path) -> tuple[int, str]:
    suffix = path.parent.name.removeprefix("user")
    return (int(suffix), str(path)) if suffix.isdigit() else (2**63 - 1, str(path))
