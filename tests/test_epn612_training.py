import json

import numpy as np

from bmiemg.train import train_from_config


def _trial(label: str, scale: float) -> dict:
    time = np.linspace(0, 2 * np.pi, 40)
    return {
        "gestureName": label,
        "groundTruthIndex": [0, 39],
        "emg": {
            f"ch{channel}": (scale * np.sin(time + channel)).tolist()
            for channel in range(1, 9)
        },
    }


def test_complete_training_pipeline_writes_artifacts(tmp_path):
    dataset = tmp_path / "dataset"
    user_path = dataset / "trainingJSON" / "user1" / "user1.json"
    user_path.parent.mkdir(parents=True)
    user_path.write_text(
        json.dumps(
            {
                "generalInfo": {"samplingFrequencyInHertz": 200},
                "userInfo": {"name": "user1"},
                "trainingSamples": {
                    "idx_1": _trial("noGesture", 1),
                    "idx_2": _trial("noGesture", 1.2),
                    "idx_3": _trial("fist", 10),
                    "idx_4": _trial("fist", 12),
                },
                "testingSamples": {
                    "idx_1": _trial("noGesture", 1.1),
                    "idx_2": _trial("fist", 11),
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results"
    config = {
        "data": {
            "root": str(dataset),
            "json_group": "trainingJSON",
            "train_section": "trainingSamples",
            "test_section": "testingSamples",
            "max_users": None,
        },
        "processing": {
            "expected_sampling_rate_hz": 200,
            "window_size_ms": 200,
            "window_step_ms": 100,
            "max_windows_per_trial": 10,
            "software_filter": {
                "enabled": False,
                "low_hz": 20,
                "high_hz": None,
                "order": 1,
            },
            "features": ["mav", "rms", "wl"],
        },
        "model": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 10000,
            "random_state": 42,
        },
        "output": {
            "directory": str(output),
            "model_filename": "model.joblib",
            "metrics_filename": "metrics.json",
            "confusion_matrix_filename": "confusion.csv",
            "feature_importance_filename": "feature_importance.csv",
            "feature_type_importance_filename": "feature_type_importance.csv",
            "feature_importance_plot_filename": "feature_importance.png",
            "feature_type_importance_plot_filename": "feature_type_importance.png",
        },
    }

    result = train_from_config(config)

    assert 0 <= result.accuracy <= 1
    assert (output / "model.joblib").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "confusion.csv").is_file()
    assert (output / "feature_importance.csv").is_file()
    assert (output / "feature_type_importance.csv").is_file()
    assert (output / "feature_importance.png").is_file()
    assert (output / "feature_type_importance.png").is_file()
