# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from dataclasses import dataclass

from .Stream import Stream



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MarkerStream(Stream):
    signal_start_time: float

    def to_annotation(self) -> mne.Annotations:
        onsets = self.time_stamps - self.signal_start_time
        descriptions = [
                str(marker[0]) if isinstance(marker, (list, tuple, np.ndarray)) else str(marker)
                for marker in self.time_series
            ]
        extras = [
            {
                "xdf_timestamp": float(timestamp),
                "marker_index": int(index),
            }
            for index, timestamp in enumerate(self.time_stamps)
        ]

        return mne.Annotations(
            onset = onsets,
            duration = np.zeros(len(onsets)),
            description = descriptions,
            extras = extras
        )
