# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from dataclasses import dataclass

from .TargetBuilder import TargetBuilder



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class OneVsAllTargetBuilder(TargetBuilder):
    def build(self, epochs: mne.Epochs, target_name: str) -> np.ndarray:
        if target_name not in epochs.event_id:
            raise ValueError(
                f"Unknown target_name '{target_name}'. "
                f"Available labels: {list(epochs.event_id.keys())}"
            )

        target_code = epochs.event_id[target_name]
        y = epochs.events[:, -1]

        return (y == target_code).astype(int)
