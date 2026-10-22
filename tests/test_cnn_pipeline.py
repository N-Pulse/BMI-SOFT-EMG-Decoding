from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bmiemg.data.epn612.loader import EMGTrial
from epn612.adaptation import BoardAdaptationConfig
from epn612.dataset import (
    EPN612DatasetConfig,
    WindowConfig,
    assign_trial_splits,
    build_epn612_datasets,
    extract_trial_windows,
    window_start_indices,
)
from epn612.split import ParticipantSource, ParticipantSplits
from cnn.cnn import TCNN
from cnn.train import checkpoint_path, parse_config
from cnn.training import TCNNConfig, build_tcnn, run_epoch, save_checkpoint


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


def test_tcnn_temporal_receptive_field_covers_roughly_50_to_100_ms():
    config = TCNNConfig(
        time_kernel=7,
        temporal_strides=(2, 2, 1),
        temporal_dilations=(1, 2, 4),
        n_filters=(2, 2, 2, 2),
    )
    model = build_tcnn(config, n_classes=6)
    convolutions = [model.block2[0], model.block3[0], model.block4[0]]

    assert config.temporal_receptive_field_samples == 127
    assert 50 <= 1000 * config.temporal_receptive_field_samples / 2000 <= 100
    assert [layer.stride for layer in convolutions] == [(2, 1), (2, 1), (1, 1)]
    assert [layer.dilation for layer in convolutions] == [(1, 1), (2, 1), (4, 1)]
    signals = torch.randn(3, 8, 400)
    features = signals.transpose(1, 2).unsqueeze(1)
    for block in (model.block1, model.block2, model.block3, model.block4):
        features = block(features)

    assert features.shape == (3, 2, 100, 1)
    assert model(signals).shape == (3, 6)


def test_single_participant_keeps_every_trials_windows_in_one_split(monkeypatch):
    import epn612.dataset as dataset_module

    source = ParticipantSource(
        participant_id="trainingJSON:user1",
        source_group="trainingJSON",
        source_user_id="user1",
        source_path=Path("user1.json"),
    )
    trials = [
        EMGTrial(
            emg=np.full((8, 20), index, dtype=np.float32),
            sampling_rate_hz=200,
            user_id="user1",
            trial_id=f"idx_{index}",
            dataset_section="trainingSamples",
            gesture="fist" if index <= 6 else "open",
            ground_truth=None,
            ground_truth_index=None,
            gesture_start=None,
            source_path=source.source_path,
        )
        for index in range(1, 13)
    ]
    monkeypatch.setattr(
        dataset_module,
        "select_participant_splits",
        lambda *args, **kwargs: ParticipantSplits(
            (source,),
            {source.participant_id: "all"},
            ("all",),
        ),
    )
    monkeypatch.setattr(
        dataset_module, "load_user_trials", lambda *args, **kwargs: iter(trials)
    )
    monkeypatch.setattr(
        dataset_module,
        "extract_trial_windows",
        lambda trial, config: [
            np.full((8, 4), int(trial.trial_id.removeprefix("idx_")), dtype=np.float32),
            np.full((8, 4), int(trial.trial_id.removeprefix("idx_")), dtype=np.float32),
        ],
    )
    config = EPN612DatasetConfig(
        root=Path("dataset"),
        participant_id=source.participant_id,
        split_ratios=(0.5, 0.25, 0.25),
        class_labels=("fist", "open"),
        windows=WindowConfig(size_samples=4, step_samples=2),
    )

    datasets = build_epn612_datasets(config, random_seed=42)

    assignments = assign_trial_splits(
        trials,
        ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
        random_seed=42,
    )
    training_values = np.array(
        [
            int(trial.trial_id.removeprefix("idx_"))
            for trial in trials
            if assignments[trial.trial_id] == "train"
        ]
    )
    np.testing.assert_allclose(datasets.normalization.mean, training_values.mean())
    np.testing.assert_allclose(
        datasets.normalization.standard_deviation,
        training_values.std(),
    )

    trial_ids_by_split = []
    for dataset in (datasets.train, datasets.validation, datasets.test):
        signals, _ = dataset.tensors
        trial_values = signals[:, 0, 0].tolist()
        assert all(
            trial_values.count(trial_value) == 2 for trial_value in set(trial_values)
        )
        trial_ids_by_split.append(set(trial_values))
    assert trial_ids_by_split[0].isdisjoint(trial_ids_by_split[1])
    assert trial_ids_by_split[0].isdisjoint(trial_ids_by_split[2])
    assert trial_ids_by_split[1].isdisjoint(trial_ids_by_split[2])


def test_trial_split_is_reproducible_and_stratified():
    trials = [
        EMGTrial(
            emg=np.zeros((8, 20), dtype=np.float32),
            sampling_rate_hz=200,
            user_id="user1",
            trial_id=f"{gesture}_{index}",
            dataset_section="trainingSamples",
            gesture=gesture,
            ground_truth=None,
            ground_truth_index=None,
            gesture_start=None,
            source_path=Path("user1.json"),
        )
        for gesture in ("fist", "open")
        for index in range(10)
    ]
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}

    first = assign_trial_splits(trials, ratios=ratios, random_seed=42)
    second = assign_trial_splits(trials, ratios=ratios, random_seed=42)

    assert first == second
    for gesture in ("fist", "open"):
        counts = {
            split: sum(
                first[trial.trial_id] == split
                for trial in trials
                if trial.gesture == gesture
            )
            for split in ratios
        }
        assert counts == {"train": 7, "validation": 2, "test": 1}


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
