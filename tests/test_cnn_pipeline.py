from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bmiemg.data.epn612.loader import EMGTrial
from scripts.cnn.cnn import TCNN
from scripts.cnn.train import checkpoint_path, parse_config
from scripts.cnn.training import run_epoch, save_checkpoint
from scripts.epn612.adaptation import BoardAdaptationConfig
from scripts.epn612.dataset import (
    EPN612DatasetConfig,
    WindowConfig,
    extract_trial_windows,
    window_start_indices,
)


def test_window_starts_are_uniformly_limited():
    config = WindowConfig(size_samples=400, step_samples=200, max_per_trial=4)

    starts = window_start_indices(0, 2000, config)

    assert starts.tolist() == [0, 400, 1000, 1600]


def test_trial_adaptation_and_windowing_are_encapsulated():
    trial = EMGTrial(
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
    config = EPN612DatasetConfig(
        root=Path("dataset"),
        class_labels=("fist",),
        adaptation=BoardAdaptationConfig(),
        windows=WindowConfig(size_samples=400, step_samples=200, max_per_trial=10),
    )

    windows = extract_trial_windows(trial, config)

    assert len(windows) == 2
    assert all(window.shape == (8, 400) for window in windows)
    assert all(window.dtype == np.float32 for window in windows)


def test_cli_overrides_preserve_the_rest_of_the_typed_config():
    config = parse_config(
        [
            "--user",
            "trainingJSON:user1",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.01",
            "--window-size",
            "200",
            "--seed",
            "7",
            "--model-name",
            "CNNtest",
            "--output-dir",
            "models",
        ]
    )

    assert config.data.participant_id == "trainingJSON:user1"
    assert config.data.windows.size_samples == 200
    assert config.training.epochs == 2
    assert config.training.batch_size == 4
    assert config.training.learning_rate == 0.01
    assert config.random_seed == 7
    assert config.model_name == "CNNtest"
    assert config.output_directory == Path("models")
    assert config.model.n_filters == (32, 32, 64, 64)
    assert checkpoint_path(config, 12) == Path("models/CNNtest_12users_100mswindow.pt")


def test_epoch_runner_handles_training_and_evaluation():
    signals = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(signals, targets), batch_size=2)
    model = nn.Linear(2, 2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    training = run_epoch(
        model,
        loader,
        loss_fn,
        torch.device("cpu"),
        optimizer=optimizer,
    )
    evaluation = run_epoch(model, loader, loss_fn, torch.device("cpu"))

    assert training.loss >= 0
    assert 0 <= training.accuracy <= 1
    assert evaluation.loss >= 0
    assert 0 <= evaluation.accuracy <= 1


def test_checkpoint_contains_reloadable_cpu_weights_and_metadata(tmp_path):
    model = TCNN(8, 6, n_filters=(2, 2, 2, 2))
    destination = tmp_path / "CNNv1_5users_200mswindow.pt"

    save_checkpoint(model, destination, {"class_labels": ("rest", "fist")})
    checkpoint = torch.load(destination, map_location="cpu", weights_only=True)

    assert checkpoint["class_labels"] == ("rest", "fist")
    assert set(checkpoint["state_dict"]) == set(model.state_dict())
    assert all(
        tensor.device.type == "cpu" for tensor in checkpoint["state_dict"].values()
    )
