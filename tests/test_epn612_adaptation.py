import json

import numpy as np

from scripts.epn612.adaptation import (
    BoardAdaptationConfig,
    adapt_emg_to_board,
    board_counts_to_normalized,
    iter_adapted_split,
    mimic_board_analog_filter,
    normalize_signed_adc,
    resample_emg,
)
from scripts.epn612.split import (
    ParticipantSource,
    SplitRecord,
    assign_participant_splits,
    build_split_records,
    summarize_split,
    write_split_manifest,
)


def test_board_adaptation_scales_resamples_filters_and_quantizes():
    config = BoardAdaptationConfig()
    source = np.zeros((8, 200), dtype=np.int8)
    source[:, 20:100] = 127

    adapted = adapt_emg_to_board(source, config)

    assert adapted.shape == (8, 2000)
    assert adapted.dtype == np.int16
    assert adapted.min() >= -2048
    assert adapted.max() <= 2047
    normalized = board_counts_to_normalized(adapted, 12)
    assert np.max(np.abs(normalized)) <= 1


def test_source_normalization_preserves_signed_fractional_scale():
    values = np.array([-128, -64, 0, 64, 127], dtype=np.int16)

    normalized = normalize_signed_adc(values, 8)

    np.testing.assert_allclose(normalized, [-1, -0.5, 0, 0.5, 127 / 128])


def test_filter_prefers_passband_over_low_and_high_stopbands():
    config = BoardAdaptationConfig()
    time = np.arange(0, 2, 1 / config.target_sampling_rate_hz)
    signal = np.stack(
        [
            np.sin(2 * np.pi * 2 * time),
            np.sin(2 * np.pi * 100 * time),
            np.sin(2 * np.pi * 900 * time),
        ]
    )

    filtered = mimic_board_analog_filter(signal, config)[:, 1000:]
    rms = np.sqrt(np.mean(filtered**2, axis=1))

    assert rms[1] > 5 * rms[0]
    assert rms[1] > 2 * rms[2]


def test_resampling_has_exact_tenfold_length():
    config = BoardAdaptationConfig()
    result = resample_emg(np.ones((8, 37), dtype=np.float32), config)
    assert result.shape == (8, 370)


def test_subject_disjoint_split_and_movement_balance(tmp_path):
    labels = ["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]
    participants = []
    for group in ("trainingJSON", "testingJSON"):
        for number in range(1, 11):
            path = tmp_path / group / f"user{number}" / f"user{number}.json"
            path.parent.mkdir(parents=True)
            training = {
                f"idx_{index}": {"gestureName": label, "emg": {}}
                for index, label in enumerate(labels, start=1)
            }
            path.write_text(
                json.dumps(
                    {
                        "generalInfo": {"samplingFrequencyInHertz": 200},
                        "trainingSamples": training,
                        "testingSamples": {},
                    }
                ),
                encoding="utf-8",
            )
            participants.append(
                ParticipantSource(
                    participant_id=f"{group}:user{number}",
                    source_group=group,
                    source_user_id=f"user{number}",
                    source_path=path,
                )
            )

    assignments = assign_participant_splits(participants, random_seed=7)
    records = build_split_records(participants, assignments)
    summary = summarize_split(records)

    assert len(assignments) == 20
    assert len(records) == 120
    assert summary["splits"]["train"]["users"] == 14
    assert summary["splits"]["test"]["users"] == 4
    assert summary["splits"]["val"]["users"] == 2
    for participant in participants:
        assert (
            len(
                {
                    row.split
                    for row in records
                    if row.participant_id == participant.participant_id
                }
            )
            == 1
        )


def test_manifest_loader_adapts_selected_trials_lazily(tmp_path):
    user_path = tmp_path / "trainingJSON" / "user1" / "user1.json"
    user_path.parent.mkdir(parents=True)
    source = np.tile(np.arange(-10, 10, dtype=int), (8, 1))
    user_path.write_text(
        json.dumps(
            {
                "generalInfo": {"samplingFrequencyInHertz": 200},
                "userInfo": {"name": "user1"},
                "trainingSamples": {
                    "idx_1": {
                        "gestureName": "fist",
                        "emg": {
                            f"ch{channel}": source[channel - 1].tolist()
                            for channel in range(1, 9)
                        },
                    }
                },
                "testingSamples": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "split.csv"
    write_split_manifest(
        [
            SplitRecord(
                participant_id="trainingJSON:user1",
                source_group="trainingJSON",
                source_user_id="user1",
                source_section="trainingSamples",
                trial_id="idx_1",
                movement="fist",
                split="train",
                source_path=str(user_path),
                source_sampling_rate_hz=200,
                target_sampling_rate_hz=2000,
                source_adc_bits=8,
                target_adc_bits=12,
                analog_filter="causal first-order 20-500 Hz band-pass",
            )
        ],
        manifest,
    )

    adapted = next(iter_adapted_split(manifest, "train"))

    assert adapted.emg_counts.shape == (8, 200)
    assert adapted.sampling_rate_hz == 2000
    assert adapted.gesture == "fist"
