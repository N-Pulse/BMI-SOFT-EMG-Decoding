import json

import numpy as np

from bmiemg.data.epn612 import iter_epn612_trials, load_user_trials


def _sample(label: str | None, length: int = 40) -> dict:
    sample = {
        "emg": {f"ch{channel}": list(range(length)) for channel in range(1, 9)},
        "groundTruth": [1] * length,
        "groundTruthIndex": [0, length - 1],
        "startPointforGestureExecution": 0,
    }
    if label is not None:
        sample["gestureName"] = label
    return sample


def _write_user(path, testing_label: str | None = "fist") -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "generalInfo": {"samplingFrequencyInHertz": 200},
                "userInfo": {"name": path.stem},
                "trainingSamples": {"idx_1": _sample("fist")},
                "testingSamples": {"idx_1": _sample(testing_label)},
            }
        ),
        encoding="utf-8",
    )


def test_load_user_trials_returns_eight_channel_float_array(tmp_path):
    path = tmp_path / "user1" / "user1.json"
    _write_user(path)

    trial = next(load_user_trials(path))

    assert trial.emg.shape == (8, 40)
    assert trial.emg.dtype == np.float32
    assert trial.sampling_rate_hz == 200
    assert trial.gesture == "fist"
    assert trial.ground_truth_index == (0, 39)


def test_dataset_iterator_supports_hidden_test_labels(tmp_path):
    path = tmp_path / "testingJSON" / "user1" / "user1.json"
    _write_user(path, testing_label=None)

    trial = next(
        iter_epn612_trials(
            tmp_path, json_group="testingJSON", section="testingSamples"
        )
    )

    assert trial.gesture is None
