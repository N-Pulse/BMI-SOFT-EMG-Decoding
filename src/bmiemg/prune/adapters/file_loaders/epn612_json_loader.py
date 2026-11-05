# ================================================================
# 0. Section: IMPORTS
# ================================================================
import json

import numpy as np

from typing import Any
from pathlib import Path
from numpy.typing import NDArray

from ...domain import DataActor
from .file_loader import FileLoader
from ...domain.actors import TrialData
from .file_loader_registry import FileLoaderRegistry

SECTION = "trainingSamples"
CHANNEL_NAMES = [f"ch{i}" for i in range(1, 9)]
MOVE_PHASE = 3
GESTURE_CODES = {
    "noGesture": 1,
    "fist": 2,
    "waveIn": 3,
    "waveOut": 4,
    "open": 5,
    "pinch": 6,
}
GROUP_OFFSETS = {"trainingJSON": 0, "testingJSON": 1000}


# ================================================================
# 1. Section: Class definition
# ================================================================
@FileLoaderRegistry.register(".json")
class Epn612JsonLoader(FileLoader):
    def load_actors(self, path: Path) -> list[DataActor]:
        return list(self._load_trials(path))

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _load_trials(self, path: Path) -> list[TrialData]:
        with path.open(encoding="utf-8") as file:
            document = json.load(file)

        general = document["generalInfo"]
        sfreq = float(general["samplingFrequencyInHertz"])
        subject_id = _subject_id(path)
        samples: dict[str, Any] = document[SECTION]

        trials = [
            TrialData(
                signal_ch=_signal(sample),
                markers=_markers(sample),
                channel_names=list(CHANNEL_NAMES),
                sfreq=sfreq,
                file_header=general,
                file_name=f"{path.stem}_{trial_id}",
                file_path=path,
                directory_structure="epn612",
                signal_type="emg",
                subject_id=subject_id,
            )
            for trial_id, sample in sorted(samples.items(), key=_trial_key)
        ]
        if not trials:
            raise ValueError(f"No {SECTION} trials in {path}")
        return trials


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _signal(sample: dict[str, Any]) -> NDArray:
    emg = sample["emg"]
    return np.stack([np.asarray(emg[name], dtype=np.float64) for name in CHANNEL_NAMES])


def _markers(sample: dict[str, Any]) -> NDArray:
    gesture = sample["gestureName"]
    if gesture not in GESTURE_CODES:
        raise ValueError(
            f"Unknown gestureName {gesture!r}; expected one of {list(GESTURE_CODES)}"
        )
    n = len(sample["emg"][CHANNEL_NAMES[0]])
    code = MOVE_PHASE * 10_000 + GESTURE_CODES[gesture]

    # noGesture trials carry no ground truth: the whole trial is the label.
    ground_truth = sample.get("groundTruth")
    if ground_truth is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(ground_truth, dtype=bool)[:n]

    markers = np.zeros((1, n), dtype=np.int64)
    markers[0, : mask.size][mask] = code
    return markers


def _subject_id(path: Path) -> int:
    user = path.parent.name.removeprefix("user")
    if not user.isdigit():
        raise ValueError(f"Expected a user<N> folder, got {path.parent.name!r}")
    group = path.parent.parent.name
    return GROUP_OFFSETS.get(group, 0) + int(user)


def _trial_key(item: tuple[str, Any]) -> int:
    suffix = item[0].rpartition("_")[2]
    return int(suffix) if suffix.isdigit() else 2**31
