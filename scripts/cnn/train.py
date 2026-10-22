"""Train the TCNN on subject-disjoint EMG-EPN-612 windows.

Run from the repository root with ``poetry run python scripts/cnn/train.py``.
The complete default experiment is intentionally declared below so a new setup
can be understood and changed without tracing constants through the codebase.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from torch import nn

from epn612.adaptation import BoardAdaptationConfig
from epn612.dataset import (
    EPN612DatasetConfig,
    WindowConfig,
    build_epn612_datasets,
)

if __package__:  # Installed package or ``python -m cnn.train``.
    from .training import (
        TCNNConfig,
        TrainingConfig,
        build_data_loaders,
        build_tcnn,
        evaluate,
        fit,
        save_checkpoint,
    )
else:  # Direct ``python scripts/cnn/train.py`` execution.
    from training import (
        TCNNConfig,
        TrainingConfig,
        build_data_loaders,
        build_tcnn,
        evaluate,
        fit,
        save_checkpoint,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Every parameter needed to reproduce an experiment."""

    data: EPN612DatasetConfig
    model: TCNNConfig
    training: TrainingConfig
    random_seed: int = 42
    device: str = "auto"
    model_name: str = "CNNv1"
    output_directory: Path = REPOSITORY_ROOT / "artifacts/cnn"

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        if not self.model_name or not all(
            character.isalnum() or character in "-_" for character in self.model_name
        ):
            raise ValueError(
                "model_name may contain only letters, numbers, '-' and '_'"
            )


# Edit this single object to define a different experiment. Common settings can
# also be overridden from the command line.
DEFAULT_CONFIG = ExperimentConfig(
    random_seed=42,
    device="auto",  # "auto", "cpu", "cuda", or "mps"
    model_name="CNNv1",
    output_directory=REPOSITORY_ROOT / "artifacts/cnn",
    data=EPN612DatasetConfig(
        root=REPOSITORY_ROOT
        / "data/raw/emg_epn_612/EMG-EPN612 Dataset/EMG-EPN612 Dataset",
        participant_id=None,
        max_participants=None,
        split_ratios=(0.70, 0.15, 0.15),  # train, validation, test
        class_labels=("noGesture", "fist", "waveIn", "waveOut", "open", "pinch"),
        adaptation=BoardAdaptationConfig(
            source_sampling_rate_hz=200.0,
            target_sampling_rate_hz=2000.0,
            source_adc_bits=8,
            target_adc_bits=12,
            highpass_hz=20.0,
            lowpass_hz=500.0,
            filter_order=1,
        ),
        windows=WindowConfig(
            size_samples=400,  # 200 ms at 2 kHz
            step_samples=200,  # 50% overlap
            max_per_trial=10,
        ),
    ),
    model=TCNNConfig(
        n_channels=8,
        time_kernel=5,
        pool_stride=1,
        n_filters=(32, 32, 64, 64),
    ),
    training=TrainingConfig(
        epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.0,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        label_smoothing=0.0,
        num_workers=0,
    ),
)


def run_experiment(config: ExperimentConfig) -> Path:
    """Train, evaluate, and export one configured experiment."""

    # print_experiment(config)
    seed_everything(config.random_seed)
    device = resolve_device(config.device)
    print(f"Device: {device}")

    # 1. EPN-612 selection -> grouped splits -> windowing -> train-fit normalization.
    datasets = build_epn612_datasets(config.data, random_seed=config.random_seed)
    print("Dataset sizes:", datasets.sizes)

    # 2. Batching (only the training loader is shuffled).
    loaders = build_data_loaders(
        datasets.train,
        datasets.validation,
        datasets.test,
        config.training,
        random_seed=config.random_seed,
    )

    # 3. Model, optimizer, and objective construction.
    model = build_tcnn(config.model, n_classes=len(datasets.class_labels)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.training.adam_betas,
        eps=config.training.adam_epsilon,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)

    # 4. Optimization with validation after each epoch.
    fit(model, loaders, optimizer, loss_fn, config.training, device)

    # 5. One final evaluation on the held-out test split.
    test_metrics = evaluate(model, loaders.test, loss_fn, device)
    print(f"test loss {test_metrics.loss:.3f} acc {test_metrics.accuracy:.2f}")

    # 6. Export CPU weights and everything needed to reconstruct the model.
    destination = checkpoint_path(config, datasets.participant_count)
    save_checkpoint(
        model,
        destination,
        {
            "format_version": 2,
            "architecture": "TCNN",
            "model_config": asdict(config.model),
            "class_labels": datasets.class_labels,
            "input": {
                "shape": (
                    "batch",
                    config.model.n_channels,
                    config.data.windows.size_samples,
                ),
                "dtype": "float32",
                "sampling_rate_hz": config.data.adaptation.target_sampling_rate_hz,
                "normalization": {
                    "type": "per_channel_standardization",
                    **asdict(datasets.normalization),
                },
            },
            "participant_count": datasets.participant_count,
            "split_sizes": datasets.sizes,
            "test_metrics": asdict(test_metrics),
            "experiment_config": _printable(asdict(config)),
        },
    )
    print(f"Model exported: {destination}")
    return destination


def parse_config(argv: list[str] | None = None) -> ExperimentConfig:
    """Apply common CLI overrides to ``DEFAULT_CONFIG``."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_CONFIG.data.root)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--user",
        default=DEFAULT_CONFIG.data.participant_id,
        help="Train one participant, e.g. trainingJSON:user1",
    )
    selection.add_argument(
        "--max-users",
        type=positive_int,
        default=DEFAULT_CONFIG.data.max_participants,
        help="Use only the first N participants",
    )
    parser.add_argument(
        "--window-size",
        type=positive_int,
        default=DEFAULT_CONFIG.data.windows.size_samples,
        help="Window length in adapted samples",
    )
    parser.add_argument(
        "--window-step",
        type=positive_int,
        default=DEFAULT_CONFIG.data.windows.step_samples,
        help="Step between windows in adapted samples",
    )
    parser.add_argument(
        "--max-windows-per-trial",
        type=positive_int,
        default=DEFAULT_CONFIG.data.windows.max_per_trial,
    )
    parser.add_argument(
        "--epochs", type=positive_int, default=DEFAULT_CONFIG.training.epochs
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_CONFIG.training.batch_size,
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=DEFAULT_CONFIG.training.learning_rate,
    )
    parser.add_argument("--device", default=DEFAULT_CONFIG.device)
    parser.add_argument("--model-name", default=DEFAULT_CONFIG.model_name)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_CONFIG.output_directory
    )
    parser.add_argument(
        "--seed", type=non_negative_int, default=DEFAULT_CONFIG.random_seed
    )
    args = parser.parse_args(argv)

    windows = replace(
        DEFAULT_CONFIG.data.windows,
        size_samples=args.window_size,
        step_samples=args.window_step,
        max_per_trial=args.max_windows_per_trial,
    )
    data = replace(
        DEFAULT_CONFIG.data,
        root=args.dataset_root,
        participant_id=args.user,
        max_participants=args.max_users,
        windows=windows,
    )
    training = replace(
        DEFAULT_CONFIG.training,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    return replace(
        DEFAULT_CONFIG,
        data=data,
        training=training,
        device=args.device,
        random_seed=args.seed,
        model_name=args.model_name,
        output_directory=args.output_dir,
    )


def print_experiment(config: ExperimentConfig) -> None:
    """Print both the pipeline composition and effective parameter values."""

    stages = (
        "select participants and create data splits",
        "adapt trials to the deployment board and extract labelled windows",
        "batch windows and construct TCNN",
        "train with Adam and cross-entropy",
        "evaluate the held-out test split",
        "export weights and deployment metadata",
    )
    print("Pipeline:")
    for index, stage in enumerate(stages, start=1):
        print(f"  {index}. {stage}")
    print("Parameters:")
    print(pformat(_printable(asdict(config)), sort_dicts=False))


def seed_everything(seed: int) -> None:
    """Seed the random generators used by this pipeline."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or select CUDA/CPU automatically."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def checkpoint_path(config: ExperimentConfig, participant_count: int) -> Path:
    """Build a descriptive checkpoint path for one experiment."""

    window_ms = 1000 * (
        config.data.windows.size_samples
        / config.data.adaptation.target_sampling_rate_hz
    )
    duration = f"{window_ms:g}".replace(".", "p")
    filename = f"{config.model_name}_{participant_count}users_{duration}mswindow.pt"
    return config.output_directory / filename


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def positive_float(value: str) -> float:
    """Parse a finite positive float for argparse."""

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return number


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""

    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return number


def _printable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _printable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_printable(item) for item in value)
    return value


def main() -> None:
    """Parse the configuration and run one experiment."""

    run_experiment(parse_config())


if __name__ == "__main__":
    main()
