# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

from pathlib import Path

from datetime import date



# ================================================================
# 1. Section: Functions
# ================================================================
def save_epochs(
    epochs: mne.Epochs | mne.EpochsArray,
    dataset_dir: Path,
    dataset_name: str,
    overwrite: bool = True,
    with_time: bool = True
) -> Path:
    # 1. Define the paths
    dataset_dir.mkdir(parents=True, exist_ok=True)
    suffix = date.today().isoformat() if with_time else ""
    epochs_path = dataset_dir / f"{dataset_name}_{suffix}_epo.fif"

    # 2. Saves the dataset
    epochs.save(epochs_path, overwrite=overwrite)

    return epochs_path
