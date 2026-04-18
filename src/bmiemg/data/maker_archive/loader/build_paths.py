# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

from pathlib import Path

#from .BIDSLoader import BIDSLoader



# ================================================================
# 1. Section: Build File and Foulder Path
# ================================================================
def build_file_name (bids) -> str:
    file_name = f"sub-{bids.subject}"
    if bids.session:
        file_name += f"_ses-{bids.session}"
    if bids.task:
        file_name += f"_task-{bids.task}"
        if bids.run:
            file_name += f"_run-{bids.run}"
    file_name += "_emg"
    file_name += bids.file_type

    return file_name

def build_foulder(bids) -> Path:
    subj_dir = bids.data_dir / f"sub-{bids.subject}"
    if bids.session:
        subj_dir = subj_dir / f"ses-{bids.session}"
    emg_dir = subj_dir / "emg"

    if not os.path.exists(emg_dir):
        raise FileNotFoundError(f"No EMG folder for subject {bids.subject}, session {bids.session}")

    return emg_dir
