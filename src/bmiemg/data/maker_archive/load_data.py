# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import re
import pandas as pd
import numpy as np
import pyxdf

from pathlib import Path
from ..DataLoader import DataLoader



# ================================================================
# 1. Section: Functions
# ================================================================
def find_bids_emg_files(
    data_dir: Path,
    subject: str,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    file_type: str = ".xdf"
) -> list:
    emg_dir = build_foulder(data_dir, subject, session)
    file_name = build_file_name(subject, session, task, run, file_type)

    files = [f for f in os.listdir(emg_dir) if re.match(file_name, f)]
    return [os.path.join(emg_dir, f) for f in files]


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def build_file_name(
    subject: str | None,
    session: str | None,
    task: str | None,
    run: str | None,
    file_type: str
) -> str:
    file_name = f"sub-{subject}"
    if session:
        file_name += f"_ses-{session}"
    if task:
        file_name += f"_task-{task}"
    if run:
        file_name += f"_run-{run}"
    file_name += "_emg"
    file_name += file_type

    return file_name

def build_foulder(
    data_dir: Path,
    subject: str,
    session: str | None = None,
):
    subj_dir = data_dir / f"sub-{subject}"
    if session:
        subj_dir = subj_dir / f"ses-{session}"
    emg_dir = subj_dir / "emg"

    if not os.path.exists(emg_dir):
        raise FileNotFoundError(f"No EMG folder for subject {subject}, session {session}")

    return emg_dir



# ================================================================
# 2. Section: Actual load
# ================================================================
def load_emg_bids(
    data_dir,
    subject,
    session=None,
    task=None,
    run=None,
    label_column="event_label"
):
    xdf_files = find_bids_emg_files(data_dir, subject, session, task, run)
    if len(xdf_files) == 0:
        raise FileNotFoundError(f"No EMG XDF found for task={task}")

    xdf_path = xdf_files[0]  # assuming one run
    base_prefix = xdf_path.replace("_emg.xdf", "")
    events_path = base_prefix + "_events.tsv"

    # --- Load EMG signal ---
    streams, header = pyxdf.load_xdf(xdf_path)

    return streams, header
