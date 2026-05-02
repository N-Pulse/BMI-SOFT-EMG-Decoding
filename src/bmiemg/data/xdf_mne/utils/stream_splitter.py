# ================================================================
# 0. Section: IMPORTS
# ================================================================
from copy import deepcopy
from typing import Any, Mapping

import numpy as np


Stream = dict[str, Any]
Channel = dict[str, Any]



# ================================================================
# 1. Section: Public API
# ================================================================
def split_stream(stream: Stream, channel_dict: Mapping[str, str]) -> tuple[Stream, Stream]:
    channels = get_channels(stream)
    channel_types = get_channel_types(channels)
    time_series = np.asarray(stream["time_series"])

    validate_stream(time_series, channels)

    eeg_idx = np.flatnonzero(channel_types == channel_dict["EEG"])
    emg_idx = np.flatnonzero(channel_types == channel_dict["EMG"])

    if eeg_idx.size == 0:
        raise ValueError("No EEG channels found.")

    if emg_idx.size == 0:
        raise ValueError("No EMG channels found.")

    eeg_stream = make_channel_subset_stream(
        stream=stream,
        indices=eeg_idx,
        stream_type="EEG",
        segment_id=2,
    )

    emg_stream = make_channel_subset_stream(
        stream=stream,
        indices=emg_idx,
        stream_type="EMG",
        segment_id=3,
    )

    return eeg_stream, emg_stream


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_channels(stream: Stream) -> list[Channel]:
    return stream["info"]["desc"][0]["channels"][0]["channels"]


def get_channel_types(channels: list[Channel]) -> np.ndarray:
    return np.array([ch["type"] for ch in channels])


def validate_stream(time_series: np.ndarray, channels: list[Channel]) -> None:
    if time_series.ndim != 2:
        raise ValueError(
            f"Expected time_series to be 2D, got shape {time_series.shape}."
        )

    n_samples, n_channels = time_series.shape

    if n_channels != len(channels):
        raise ValueError(
            "Mismatch between time_series channels and metadata channels: "
            f"time_series has {n_channels} channels, "
            f"metadata has {len(channels)} channels."
        )


def make_channel_subset_stream(
    stream: Stream,
    indices: np.ndarray,
    stream_type: str,
    segment_id: int,
) -> Stream:
    out = deepcopy(stream)

    original_channels = get_channels(stream)

    out["info"]["type"] = [stream_type]
    out["info"]["channel_count"] = [str(len(indices))]
    out["info"]["segment_id"] = segment_id

    out["info"]["desc"][0]["channels"][0]["channels"] = [
        deepcopy(original_channels[i]) for i in indices
    ]

    out["time_series"] = np.asarray(stream["time_series"])[:, indices]

    return out
