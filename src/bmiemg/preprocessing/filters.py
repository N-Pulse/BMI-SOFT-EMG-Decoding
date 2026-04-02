# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pandas as pd

from scipy import signal
from typing import cast



# ================================================================
# 1. Section: Filters
# ================================================================
def notch_filter(
    df: pd.DataFrame,
    fs: int = 1000,
    freq: float = 50.0,
    q: float = 30.0,
    single: bool = False
) -> pd.DataFrame:
    """Remove power line interference at 50 Hz (or 60 Hz for US)"""
    b, a = signal.iirnotch(freq, q, fs)

    filtered_df = df.copy()
    for col in df.columns:
        # Skip non-time-series columns
        if col == 'window_index' or col == 'label':
            continue

        if not single:
            # Apply filter only to columns that contain arrays
            filtered_df[col] = df[col].apply(lambda x: signal.filtfilt(b, a, x) if isinstance(x, (list, np.ndarray)) and len(x) > 1 else x)
        else:
            filtered_df[col] = signal.filtfilt(b, a, df[col])

    return filtered_df

def passband_filter(
    df: pd.DataFrame,
    fs: int = 1000,
    lowcut: float = 20.0,
    highcut: float = 300.0,
    order: int = 4,
    single: bool = False
) -> pd.DataFrame:
    """Bandpass filter for EMG signals (20-450 Hz)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cast(tuple, signal.butter(order, [low, high], btype='band'))

    filtered_df = df.copy()
    for col in df.columns:
        # Skip non-time-series columns
        if col == 'window_index' or col == 'label':
            continue

        if not single:
            # Apply filter only to columns that contain arrays
            filtered_df[col] = df[col].apply(lambda x: signal.filtfilt(b, a, x) if isinstance(x, (list, np.ndarray)) and len(x) > 1 else x)
        else:
            filtered_df[col] = signal.filtfilt(b, a, df[col])

    return filtered_df
