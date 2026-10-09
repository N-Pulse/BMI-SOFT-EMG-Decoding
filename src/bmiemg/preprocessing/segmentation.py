"""Offline conversion of annotated trials to fixed-size labelled windows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from bmiemg.data.epn612 import EMGTrial


FloatArray: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class LabelledEMGWindow:
    signal: FloatArray
    gesture: str
    user_id: str
    trial_id: str
    window_index: int
    start_sample: int


def labelled_interval(trial: EMGTrial) -> tuple[int, int]:
    """Return a half-open active interval using annotations only offline."""

    sample_count = trial.emg.shape[1]
    if trial.ground_truth_index is not None:
        start, inclusive_end = trial.ground_truth_index
        return max(0, start), min(sample_count, inclusive_end + 1)
    if trial.ground_truth is not None:
        active = np.flatnonzero(trial.ground_truth != 0)
        if active.size:
            return int(active[0]), min(sample_count, int(active[-1]) + 1)
    return 0, sample_count


def iter_trial_windows(
    trial: EMGTrial,
    window_size_samples: int,
    window_step_samples: int,
    max_windows: int | None = None,
) -> Iterator[LabelledEMGWindow]:
    """Yield fixed windows, evenly limiting long trials when requested."""

    if trial.gesture is None:
        raise ValueError(f"Unlabelled trial {trial.user_id}:{trial.trial_id}")
    if window_size_samples <= 0 or window_step_samples <= 0:
        raise ValueError("Window size and step must be positive")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive or None")

    first, stop = labelled_interval(trial)
    if stop - first < window_size_samples:
        return
    starts = np.arange(
        first, stop - window_size_samples + 1, window_step_samples, dtype=int
    )
    if max_windows is not None and len(starts) > max_windows:
        indexes = np.linspace(0, len(starts) - 1, max_windows, dtype=int)
        starts = starts[indexes]

    for index, start in enumerate(starts.tolist()):
        yield LabelledEMGWindow(
            signal=np.asarray(
                trial.emg[:, start : start + window_size_samples], dtype=np.float32
            ),
            gesture=trial.gesture,
            user_id=trial.user_id,
            trial_id=trial.trial_id,
            window_index=index,
            start_sample=start,
        )
