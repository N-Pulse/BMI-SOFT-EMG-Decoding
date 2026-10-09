"""EMG window transformation shared by training and online inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

from .list_features import FREQ_FEATURE_FUNCTIONS, TIME_FEATURE_FUNCTIONS

FloatArray: TypeAlias = NDArray[np.float32]
FeatureFunction: TypeAlias = Callable[..., np.ndarray]

TIME_FEATURES: Mapping[str, FeatureFunction] = {
    function.__name__: function for function in TIME_FEATURE_FUNCTIONS
}
FREQUENCY_FEATURES: Mapping[str, FeatureFunction] = {
    function.__name__: function for function in FREQ_FEATURE_FUNCTIONS
}
FEATURE_FUNCTIONS: Mapping[str, FeatureFunction] = {
    **TIME_FEATURES,
    **FREQUENCY_FEATURES,
}
ALL_FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_FUNCTIONS)


@dataclass(frozen=True, slots=True)
class EMGProcessingConfig:
    """All parameters needed to reproduce model inputs online."""

    sampling_rate_hz: float
    window_size_samples: int
    feature_names: tuple[str, ...]
    software_filter_enabled: bool = True
    filter_low_hz: float = 20.0
    filter_high_hz: float | None = None
    filter_order: int = 1
    channel_count: int = 8

    def __post_init__(self) -> None:
        if self.sampling_rate_hz <= 0 or self.window_size_samples <= 0:
            raise ValueError("Sampling rate and window size must be positive")
        if self.channel_count != 8:
            raise ValueError("The decoder requires exactly eight EMG channels")
        if not self.feature_names:
            raise ValueError("At least one feature must be selected")
        unknown = set(self.feature_names) - set(FEATURE_FUNCTIONS)
        if unknown:
            raise ValueError(f"Unknown features: {sorted(unknown)}")
        if self.software_filter_enabled:
            nyquist = self.sampling_rate_hz / 2
            if not 0 < self.filter_low_hz < nyquist:
                raise ValueError(f"Low cutoff must be below Nyquist ({nyquist} Hz)")
            if self.filter_high_hz is not None and not (
                self.filter_low_hz < self.filter_high_hz < nyquist
            ):
                raise ValueError(
                    f"High cutoff must be between low cutoff and Nyquist ({nyquist} Hz)"
                )
            if self.filter_order <= 0:
                raise ValueError("Filter order must be positive")


def milliseconds_to_samples(milliseconds: float, sampling_rate_hz: float) -> int:
    if milliseconds <= 0 or sampling_rate_hz <= 0:
        raise ValueError("Duration and sampling rate must be positive")
    return max(1, round(milliseconds * sampling_rate_hz / 1000))


def validate_emg_windows(
    windows: np.ndarray, config: EMGProcessingConfig
) -> FloatArray:
    """Accept one ``(8, samples)`` window or a batch of windows."""

    array = np.asarray(windows, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    expected = (config.channel_count, config.window_size_samples)
    if array.ndim != 3 or array.shape[1:] != expected or array.shape[0] == 0:
        raise ValueError(
            f"Expected (n_windows, {expected[0]}, {expected[1]}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("EMG contains NaN or infinite values")
    return array


def filter_emg_windows(windows: np.ndarray, config: EMGProcessingConfig) -> FloatArray:
    """Apply a causal filter which can also be used on online windows.

    EMG-EPN-612 is sampled at 200 Hz, so it cannot represent the production
    board's 500 Hz upper cutoff. With ``filter_high_hz=None``, only the
    first-order 20 Hz high-pass stage is applied. Production code should disable
    this software filter when it receives the board's analog-filtered signal.
    """

    array = validate_emg_windows(windows, config)
    if not config.software_filter_enabled:
        return array

    if config.filter_high_hz is None:
        cutoff: float | list[float] = config.filter_low_hz
        filter_type = "highpass"
    else:
        cutoff = [config.filter_low_hz, config.filter_high_hz]
        filter_type = "bandpass"
    sos = butter(
        config.filter_order,
        cutoff,
        btype=filter_type,
        fs=config.sampling_rate_hz,
        output="sos",
    )
    return np.asarray(sosfilt(sos, array, axis=-1), dtype=np.float32)


def extract_emg_features(
    windows: np.ndarray, config: EMGProcessingConfig
) -> FloatArray:
    """Produce feature-major, then channel-major scalar model inputs."""

    filtered = filter_emg_windows(windows, config)
    blocks = []
    for name in config.feature_names:
        function = FEATURE_FUNCTIONS[name]
        if name in FREQUENCY_FEATURES:
            blocks.append(function(filtered, config.sampling_rate_hz))
        else:
            blocks.append(function(filtered))
    features = np.concatenate(blocks, axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced non-finite values")
    return np.asarray(features, dtype=np.float32)


def expanded_feature_names(config: EMGProcessingConfig) -> tuple[str, ...]:
    return tuple(
        f"{feature}_ch{channel}"
        for feature in config.feature_names
        for channel in range(1, config.channel_count + 1)
    )
