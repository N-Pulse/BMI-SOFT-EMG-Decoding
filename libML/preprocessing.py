import os
import yaml
import numpy as np
from scipy.signal import butter, filtfilt

# ---------- FILTERS ----------
def bandpass_filter(data, fs=1000, low=20, high=450, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, data, axis=0)

# ---------- BASELINE HANDLING ----------
def compute_baseline_stats(resting_emg, save_path):
    """
    Compute mean and std for each EMG channel from resting data.
    Saves to YAML (portable) or NPY (compact).
    """
    mean = np.mean(resting_emg, axis=0)
    std = np.std(resting_emg, axis=0)

    stats = {"mean": mean.tolist(), "std": std.tolist()}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.safe_dump(stats, f)

    print(f"[INFO] Baseline saved to {save_path}")
    return mean, std


def load_baseline_stats(path):
    """
    Load per-channel mean/std normalization constants.
    """
    with open(path, "r") as f:
        stats = yaml.safe_load(f)
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    return mean, std


# ---------- NORMALIZATION ----------
def apply_preprocessing(emg, fs=1000, baseline_path=None):
    """
    Apply preprocessing:
    1. Bandpass filter
    2. Rectify
    3. Normalize using baseline stats (if provided)
    """
    filtered = bandpass_filter(emg, fs)
    rectified = np.abs(filtered)

    if baseline_path is not None:
        mean, std = load_baseline_stats(baseline_path)
        normed = (rectified - mean) / std
    else:
        # Fallback normalization (e.g., dataset-wide)
        print(f"Error: couldn't find baseline stats at the following path: {baseline_path}")
        normed = (rectified - np.mean(rectified, axis=0)) / np.std(rectified, axis=0)

    return normed
