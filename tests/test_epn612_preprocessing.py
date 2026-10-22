from pathlib import Path

import numpy as np

from bmiemg.data.epn612 import EMGTrial
from bmiemg.preprocessing import (
    ALL_FEATURE_NAMES,
    ChannelNormalizationStats,
    EMGProcessingConfig,
    expanded_feature_names,
    extract_emg_features,
    fit_channel_normalization,
    iter_trial_windows,
    normalize_emg_channels,
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


def test_channel_normalization_is_fitted_per_channel():
    training = np.array(
        [
            [[0, 2], [10, 14]],
            [[2, 4], [14, 18]],
        ],
        dtype=np.float32,
    )

    stats = fit_channel_normalization(training, epsilon=1e-6)
    normalized = normalize_emg_channels(training, stats)

    np.testing.assert_allclose(stats.mean, [2, 14])
    np.testing.assert_allclose(stats.standard_deviation, [np.sqrt(2), np.sqrt(8)])
    np.testing.assert_allclose(normalized.mean(axis=(0, 2)), 0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=(0, 2)), 1, atol=1e-5)


def test_saved_training_statistics_are_reused_for_new_windows():
    stats = ChannelNormalizationStats(
        mean=(1.0, 10.0),
        standard_deviation=(2.0, 5.0),
        epsilon=1e-6,
    )
    unseen = np.array([[3.0, -1.0], [15.0, 5.0]], dtype=np.float32)

    normalized = normalize_emg_channels(unseen, stats)

    np.testing.assert_allclose(normalized, [[1.0, -1.0], [1.0, -1.0]], atol=1e-6)
