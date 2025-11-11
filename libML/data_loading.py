import os
import re
import mne
import pandas as pd
import numpy as np

def find_bids_emg_files(bids_root, subject, session=None, task=None):
    """
    Return list of (edf_path, events_path, json_path) for one subject/session/task.
    """
    subj_dir = os.path.join(bids_root, f"sub-{subject}")
    if session:
        subj_dir = os.path.join(subj_dir, f"ses-{session}")
    if task:
        subj_dir = os.path.join(subj_dir, f"task-{task}")

    emg_dir = os.path.join(subj_dir, "_emg")
    print(emg_dir)
    return 
    if not os.path.exists(emg_dir):
        raise FileNotFoundError(f"No EMG folder for subject {subject}, session {session}")

    pattern = f"sub-{subject}"
    if session:
        pattern += f"_ses-{session}"
    if task:
        pattern += f"_task-{task}"
    pattern += ".*_emg\\.edf$"

    files = [f for f in os.listdir(emg_dir) if re.match(pattern, f)]
    return [os.path.join(emg_dir, f) for f in files]


def load_emg_bids(bids_root, subject, session, task, label_column="event_label"):
    """
    Load EMG data and labels from a BIDS-compliant directory.
    Returns: X (samples x channels), y (labels)
    """
    edf_files = find_bids_emg_files(bids_root, subject, session, task)
    if len(edf_files) == 0:
        raise FileNotFoundError(f"No EMG EDF found for task={task}")

    edf_path = edf_files[0]  # assuming one run
    base_prefix = edf_path.replace("_emg.edf", "")
    events_path = base_prefix + "_events.tsv"

    # --- Load EMG signal ---
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    emg_data = raw.get_data().T  # shape: (n_samples, n_channels)
    sfreq = raw.info["sfreq"]

    # --- Load events ---
    events_df = pd.read_csv(events_path, sep="\t")

    if label_column not in events_df.columns:
        raise ValueError(f"Missing {label_column} in events.tsv")

    # Align EMG and events (simple approach: expand labels per time window)
    y = events_df[label_column].to_numpy()

    return emg_data, y, sfreq
