# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from functools import partial
from scipy.ndimage import uniform_filter1d



# ================================================================
# 1. Section: Functions
# ================================================================
def get_envelop(
    epochs: mne.Epochs,
    window_s: float = 0.100,
    picks: str | list[str] | None = "all",
    mode: str = "reflect"
) -> mne.Epochs:
    # 1. Load the data
    env = epochs.copy().load_data()
    sfreq = env.info["sfreq"]
    window_samples = max(1, int(round(window_s * sfreq)))

    # 2. Apply the moving window
    rms_func = partial(
        moving_rms,
        window_samples=window_samples,
        mode=mode
    )
    env.apply_function(
        rms_func,
        picks=picks,
        channel_wise=True,
    )

    return env


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def moving_rms(x: np.ndarray, window_samples: int, mode: str) -> np.ndarray:
    return np.sqrt(
        uniform_filter1d(
            x ** 2,
            size=window_samples,
            mode=mode,
        )
    )
