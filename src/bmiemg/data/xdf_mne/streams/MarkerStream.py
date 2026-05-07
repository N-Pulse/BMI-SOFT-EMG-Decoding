# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np
import pandas as pd

from dataclasses import dataclass
from .EEGStream import EEGStream



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MarkerStream:
    time_series: np.ndarray
    time_stamps: np.ndarray
    info: dict
    footer: dict

    def to_annotations(self, eeg_stream: EEGStream) -> mne.Annotations:
        marker_df = _make_marker_dataframe(
            time_series = self.time_series,
            time_stamps = self.time_stamps,
            eeg_start_time = eeg_stream.time_stamps[0],
        )

        descriptions = [
            f"{row.marker_code}: {row.marker_label}"
            for row in marker_df.itertuples()
        ]

        return mne.Annotations(
            onset=marker_df["onset"].to_numpy(),
            duration=np.zeros(len(marker_df)),
            description=descriptions,
        )


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _make_marker_dataframe(
    time_series: np.ndarray,
    time_stamps: np.ndarray,
    eeg_start_time: float,
) -> pd.DataFrame:
    marker_codes = time_series.squeeze().astype(int)

    if marker_codes.ndim == 0:
        marker_codes = np.array([int(marker_codes)])

    relative_times = time_stamps - eeg_start_time

    return pd.DataFrame({
        "absolute_time": time_stamps,
        "onset": relative_times,
        "marker_code": marker_codes,
    })


def _add_annotations(
    raw: mne.io.Raw,
    marker_df: pd.DataFrame,
) -> mne.io.Raw:
    valid = (
        (marker_df["onset"] >= 0)
        & (marker_df["onset"] <= raw.times[-1])
    )

    if not valid.all():
        n_invalid = (~valid).sum()
        print(f"Warning: dropping {n_invalid} markers outside EEG time range.")

    valid_markers = marker_df.loc[valid].copy()

    descriptions = [
        f"{row.marker_code}: {row.marker_label}"
        for row in valid_markers.itertuples()
    ]

    annotations = mne.Annotations(
        onset=valid_markers["onset"].to_numpy(),
        duration=np.zeros(len(valid_markers)),
        description=descriptions,
    )

    raw.set_annotations(annotations)

    return raw
