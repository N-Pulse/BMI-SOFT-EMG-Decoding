from pathlib import Path

import numpy as np

from bmiemg.data.epn612 import EMGTrial
from bmiemg.preprocessing import (
    ALL_FEATURE_NAMES,
    EMGProcessingConfig,
    expanded_feature_names,
    extract_emg_features,
    iter_trial_windows,
)


def _trial() -> EMGTrial:
    return EMGTrial(
        emg=np.tile(np.arange(100, dtype=np.float32), (8, 1)),
        sampling_rate_hz=200,
        user_id="user1",
        trial_id="idx_1",
        dataset_section="trainingSamples",
        gesture="fist",
        ground_truth=None,
        ground_truth_index=(20, 79),
        gesture_start=20,
        source_path=Path("user1.json"),
    )


def test_segmentation_uses_ground_truth_interval():
    windows = list(iter_trial_windows(_trial(), 40, 20))

    assert [window.start_sample for window in windows] == [20, 40]
    assert all(window.signal.shape == (8, 40) for window in windows)
    assert all(window.gesture == "fist" for window in windows)


def test_feature_shape_and_names_are_stable():
    config = EMGProcessingConfig(
        sampling_rate_hz=200,
        window_size_samples=40,
        feature_names=("mav", "rms", "wl", "zc", "ssc"),
        software_filter_enabled=False,
    )
    windows = np.stack(
        [window.signal for window in iter_trial_windows(_trial(), 40, 20)]
    )

    features = extract_emg_features(windows, config)

    assert features.shape == (2, 40)
    assert len(expanded_feature_names(config)) == 40
    assert expanded_feature_names(config)[0] == "mav_ch1"
    assert expanded_feature_names(config)[-1] == "ssc_ch8"


def test_all_scalar_features_are_finite_and_expand_per_channel():
    config = EMGProcessingConfig(
        sampling_rate_hz=200,
        window_size_samples=40,
        feature_names=ALL_FEATURE_NAMES,
        software_filter_enabled=False,
    )
    windows = np.stack(
        [window.signal for window in iter_trial_windows(_trial(), 40, 20)]
    )

    features = extract_emg_features(windows, config)

    assert len(ALL_FEATURE_NAMES) == 24
    assert features.shape == (2, 24 * 8)
    assert np.isfinite(features).all()
    assert expanded_feature_names(config)[-1] == "peak_freq_ch8"
