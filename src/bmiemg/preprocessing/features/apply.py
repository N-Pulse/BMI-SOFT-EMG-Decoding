# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_emg_features(
    signal: np.ndarray,
    sfreq: float,
    time_features: list,
    freq_features: list,
) -> np.ndarray:
    feature_values = []

    for feature_func in time_features:
        val = feature_func(signal)
        if val.ndim != 2:
            raise ValueError(
                f"{feature_func.__name__} returned shape {val.shape}, "
                "expected (n_epochs, n_channels)."
            )
        feature_values.append(val)

    for feature_func in freq_features:
        val = feature_func(signal, sfreq)
        if val.ndim != 2:
            raise ValueError(
                f"{feature_func.__name__} returned shape {val.shape}, "
                "expected (n_epochs, n_channels)."
            )
        feature_values.append(val)

    features = np.concatenate(feature_values, axis=1)

    return features
