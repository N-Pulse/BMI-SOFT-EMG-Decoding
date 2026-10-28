"""Small, dataset-independent PyTorch training utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

if __package__:  # Installed package or ``python -m cnn.train``.
    from .cnn import TCNN
else:  # Direct import while running ``python scripts/cnn/train.py``.
    from cnn import TCNN


@dataclass(frozen=True, slots=True)
class TCNNConfig:
    """Parameters controlling the TCNN architecture."""

    n_channels: int = 8
    time_kernel: int = 7
    temporal_strides: tuple[int, int, int] = (2, 2, 1)
    temporal_dilations: tuple[int, int, int] = (1, 2, 4)
    n_filters: tuple[int, int, int, int] = (32, 32, 64, 64)

    def __post_init__(self) -> None:
        if self.n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if self.time_kernel <= 0 or self.time_kernel % 2 == 0:
            raise ValueError("time_kernel must be a positive odd integer")
        if len(self.temporal_strides) != 3 or any(
            value <= 0 for value in self.temporal_strides
        ):
            raise ValueError("temporal_strides must contain three positive integers")
        if len(self.temporal_dilations) != 3 or any(
            value <= 0 for value in self.temporal_dilations
        ):
            raise ValueError("temporal_dilations must contain three positive integers")
        if len(self.n_filters) != 4 or any(value <= 0 for value in self.n_filters):
            raise ValueError("n_filters must contain four positive integers")

    @property
    def temporal_receptive_field_samples(self) -> int:
        """Return the input span seen by one final temporal feature."""

        receptive_field = 1
        input_spacing = 1
        for stride, dilation in zip(
            self.temporal_strides,
            self.temporal_dilations,
            strict=True,
        ):
            receptive_field += (self.time_kernel - 1) * dilation * input_spacing
            input_spacing *= stride
        return receptive_field


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization and data-loader parameters."""

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    label_smoothing: float = 0.0
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if len(self.adam_betas) != 2 or any(
            not math.isfinite(value) or not 0 <= value < 1 for value in self.adam_betas
        ):
            raise ValueError("adam_betas must contain two values in [0, 1)")
        if not math.isfinite(self.adam_epsilon) or self.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be positive")
        if not math.isfinite(self.label_smoothing) or not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")


@dataclass(frozen=True, slots=True)
class Metrics:
    """Mean loss and accuracy for one complete dataset pass."""

    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class DataLoaders:
    """Loaders used by the train, validation, and test stages."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_tcnn(config: TCNNConfig, n_classes: int) -> TCNN:
    """Construct a TCNN directly from its visible configuration."""

    return TCNN(
        n_channels=config.n_channels,
        n_classes=n_classes,
        time_kernel=config.time_kernel,
        temporal_strides=config.temporal_strides,
        temporal_dilations=config.temporal_dilations,
        n_filters=config.n_filters,
    )


def build_data_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    config: TrainingConfig,
    *,
    random_seed: int,
) -> DataLoaders:
    """Create loaders, shuffling only the training dataset."""

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
    }
    return DataLoaders(
        train=DataLoader(
            train_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(random_seed),
            **common,
        ),
        validation=DataLoader(validation_dataset, **common),
        test=DataLoader(test_dataset, **common),
    )


def fit(
    model: nn.Module,
    loaders: DataLoaders,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> list[tuple[Metrics, Metrics]]:
    """Train the model and report train/validation metrics for each epoch."""

    history: list[tuple[Metrics, Metrics]] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders.train,
            loss_fn,
            device,
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model,
            loaders.validation,
            loss_fn,
            device,
        )
        history.append((train_metrics, validation_metrics))
        print(
            f"epoch {epoch}: "
            f"train loss {train_metrics.loss:.3f} "
            f"acc {train_metrics.accuracy:.2f} | "
            f"val loss {validation_metrics.loss:.3f} "
            f"acc {validation_metrics.accuracy:.2f}"
        )
    return history


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Metrics:
    """Evaluate an already-trained model without updating it."""

    return run_epoch(model, loader, loss_fn, device)


def save_checkpoint(
    model: nn.Module,
    destination: Path,
    metadata: dict[str, Any],
) -> Path:
    """Save CPU model weights and deployment metadata in one checkpoint."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    torch.save({**metadata, "state_dict": state_dict}, destination)
    return destination


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    optimizer: Optimizer | None = None,
) -> Metrics:
    """Run one training or evaluation pass, depending on ``optimizer``."""

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            logits = model(signals)
            loss = loss_fn(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = len(targets)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()

    if total_samples == 0:
        raise ValueError("Cannot run an epoch on an empty dataset")
    return Metrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )
