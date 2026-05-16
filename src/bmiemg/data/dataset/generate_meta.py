# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne
import json
import platform

import numpy as np
import pandas as pd

from typing import Any
from pathlib import Path
from datetime import date, datetime



# ================================================================
# 1. Section: Dataset saving
# ================================================================
def save_epochs_metadata(
    epochs: mne.Epochs,
    dataset_name: str,
    output_dir: Path,
    v1_files: list[Path],
    v2_files: list[Path],
    ignored_v1: set[str],
    ignored_v2: set[str],
    cutoff_date: date,
    root_path: Path,
    data_path: Path,
    extra: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{dataset_name}_metadata.json"
    md_path = output_dir / f"{dataset_name}_README.md"

    metadata = build_epochs_metadata(
        epochs=epochs,
        dataset_name=dataset_name,
        v1_files=v1_files,
        v2_files=v2_files,
        ignored_v1=ignored_v1,
        ignored_v2=ignored_v2,
        cutoff_date=cutoff_date,
        root_path=root_path,
        data_path=data_path,
        extra=extra,
    )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = make_dataset_readme(metadata)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    return metadata



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _json_safe(value) -> Any:
    """Convert NumPy / Path / date objects into JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    return value

def build_epochs_metadata(
    epochs: mne.Epochs,
    dataset_name: str,
    v1_files: list[Path],
    v2_files: list[Path],
    ignored_v1: set[str],
    ignored_v2: set[str],
    cutoff_date: date,
    root_path: Path,
    data_path: Path,
    extra: dict | None = None,
) -> dict:
    """Build a JSON-serialisable metadata dictionary for an epoch dataset."""

    event_counts = {
        event_name: int(np.sum(epochs.events[:, -1] == event_code))
        for event_name, event_code in epochs.event_id.items()
    }

    metadata = {
        "dataset_name": dataset_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Naive grouped EMG epoch dataset built from XDF recordings.",
        "source": {
            "root": root_path,
            "data_root": data_path,
            "cutoff_date": cutoff_date,
            "v1_protocol": {
                "n_files": len(v1_files),
                "files": [file.name for file in v1_files],
                "ignored_files": sorted(ignored_v1),
            },
            "v2_protocol": {
                "n_files": len(v2_files),
                "files": [file.name for file in v2_files],
                "ignored_files": sorted(ignored_v2),
            },
        },
        "epochs": {
            "n_epochs": len(epochs),
            "n_channels": len(epochs.ch_names),
            "n_times": len(epochs.times),
            "shape": epochs.get_data(copy=False).shape,
            "sfreq": epochs.info["sfreq"],
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
            "duration_s": float(epochs.tmax - epochs.tmin),
            "baseline": epochs.baseline,
        },
        "channels": {
            "names": epochs.ch_names,
            "types": epochs.get_channel_types(),
        },
        "events": {
            "event_id": epochs.event_id,
            "event_counts": event_counts,
            "total_events": int(len(epochs.events)),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "notes": [
            "This file contains metadata only, not signal data.",
            "The corresponding MNE Epochs object is saved as a FIF file.",
            "Class labels are stored in epochs.event_id.",
            "Labels were grouped using the active SignalPartitioner trigger maps.",
        ],
    }

    if extra is not None:
        metadata["extra"] = extra

    return _json_safe(metadata)

def make_dataset_readme(metadata: dict) -> str:
    """Create a compact Markdown README from dataset metadata."""

    event_rows = "\n".join(
        f"| {name} | {code} | {metadata['events']['event_counts'].get(name, 0)} |"
        for name, code in metadata["events"]["event_id"].items()
    )

    channel_rows = "\n".join(
        f"| {name} | {ch_type} |"
        for name, ch_type in zip(
            metadata["channels"]["names"],
            metadata["channels"]["types"],
        )
    )

    return f"""# {metadata["dataset_name"]}

    ## Description

    {metadata["description"]}

    ## Created

    {metadata["created_at"]}

    ## Epoch summary

    | Field | Value |
    |---|---:|
    | Number of epochs | {metadata["epochs"]["n_epochs"]} |
    | Number of channels | {metadata["epochs"]["n_channels"]} |
    | Number of time samples | {metadata["epochs"]["n_times"]} |
    | Sampling frequency | {metadata["epochs"]["sfreq"]} Hz |
    | tmin | {metadata["epochs"]["tmin"]} s |
    | tmax | {metadata["epochs"]["tmax"]} s |
    | Duration | {metadata["epochs"]["duration_s"]} s |

    ## Events

    | Label | Code | Count |
    |---|---:|---:|
    {event_rows}

    ## Channels

    | Channel | Type |
    |---|---|
    {channel_rows}

    ## Source files

    | Protocol | Number of files |
    |---|---:|
    | V1 | {metadata["source"]["v1_protocol"]["n_files"]} |
    | V2 | {metadata["source"]["v2_protocol"]["n_files"]} |

    Cutoff date: `{metadata["source"]["cutoff_date"]}`

    ## Ignored files

    ### V1

    {chr(10).join(f"- `{file}`" for file in metadata["source"]["v1_protocol"]["ignored_files"]) or "- None"}

    ### V2

    {chr(10).join(f"- `{file}`" for file in metadata["source"]["v2_protocol"]["ignored_files"]) or "- None"}

    ## Software

    | Package | Version |
    |---|---|
    | Python | {metadata["software"]["python"]} |
    | MNE | {metadata["software"]["mne"]} |
    | NumPy | {metadata["software"]["numpy"]} |
    | pandas | {metadata["software"]["pandas"]} |

    ## Notes

    {chr(10).join(f"- {note}" for note in metadata["notes"])}
    """

"""
metadata = save_epochs_dataset(
    epochs=grouped_epochs,
    dataset_name="naive_grouped_emg_v1_v2",
    output_dir=OUTPUT_DIR,
    v1_files=v1_files,
    v2_files=v2_files,
    ignored_v1=SESSIONS_TO_IGNORE_V1,
    ignored_v2=SESSIONS_TO_IGNORE_V2,
    cutoff_date=CUTOFF_DATE,
    overwrite=True,
    extra={
        "dataset_type": "MNE Epochs",
        "task": "grouped EMG movement classification",
        "labels": list(grouped_epochs.event_id.keys()),
    },
)
"""
