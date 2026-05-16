"""
This builds the dataset that takes both versions of the in-house campus
biotech/chuv acquisitions. The labels are grouped as wrist, grasp or pinch.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import io
import mne
import contextlib

from pathlib import Path
from datetime import date

from bmiemg.data.convert import session_load, ChannelSplitter
from bmiemg.data.dataset import save_epochs
from bmiemg.data.utils import get_xdf_paths
from bmiemg.data.epoch import (
    SignalPartitioner,
    V1_TRIGGER_MAP,
    V2_TRIGGER_MAP,
)


# ================================================================
# 1. Section: INPUTS
# ================================================================
# Paths
ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data" / "bids"

# Version relative inputs
SESSIONS_TO_IGNORE_V1: set = set([
    "sub-05_ses-04_task-Down_run-01_raw.xdf"
])
SESSIONS_TO_IGNORE_V2: set = set([])
CUTOFF_DATE = date(2025, 11, 21)

DATASET_PATH: Path = ROOT / "data" / "dataset"
DATASET_NAME: str = "naive_archive_eeg"



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def load_emg_sessions(
    splitter: ChannelSplitter,
    list_of_files: list[Path]
) -> list[mne.io.RawArray]:
    emg_signals = []
    for file in list_of_files:
        # 1. Load the session
        print(f"Loading {file}")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            session = session_load(file)
            bio_signal = splitter.split(session)
            signal = bio_signal.attach_annotations()
            emg_signals.append(signal["EEG"])

        del(bio_signal)
        del(signal)
        del(session)

    return emg_signals

def extract_epochs(emg_signals: list, partinioner: SignalPartitioner):
    epochs_group = []
    for emg_signal in emg_signals:
        epochs_group.append(partinioner.partition(emg_signal))

    merged_epochs = mne.concatenate_epochs(epochs_group)

    return merged_epochs



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Finding the files
    v1_files = get_xdf_paths(SESSIONS_TO_IGNORE_V1, "before", DATA, CUTOFF_DATE)
    v2_files = get_xdf_paths(SESSIONS_TO_IGNORE_V2, "after", DATA, CUTOFF_DATE)
    total_nr_files = len(v1_files) + len(v2_files)
    print(
        f"Found {len(v1_files)} files from V1 protocol and "
        f"found {len(v2_files)} from V2 protocol (total {total_nr_files})\n"
    )

    # 2. Loads the files
    biotech_splitter = ChannelSplitter()
    emg_signals_v1 = load_emg_sessions(biotech_splitter, v1_files)
    emg_signals_v2 = load_emg_sessions(biotech_splitter, v2_files)
    total_nr_loaded_files = len(emg_signals_v1) + len(emg_signals_v2)
    print(
        f"Finished loading the {total_nr_loaded_files} from the original "
        f"{total_nr_files} paths (success = "
        f"{(total_nr_loaded_files/total_nr_files * 100)}\n"
    )

    # 3. Extracts the epochs
    partinioner_v1 = SignalPartitioner(V1_TRIGGER_MAP)
    partinioner_v2 = SignalPartitioner(V2_TRIGGER_MAP)
    epochs_v1 = extract_epochs(emg_signals_v1, partinioner_v1)
    epochs_v2 = extract_epochs(emg_signals_v2, partinioner_v2)

    # 4, Organize into target ones
    grouped_v1_epochs = partinioner_v1.group(epochs_v1)
    grouped_v2_epochs = partinioner_v2.group(epochs_v2)
    grouped_epochs = mne.concatenate_epochs([grouped_v1_epochs, grouped_v2_epochs])

    # 5. Save the data
    p = save_epochs(grouped_epochs, DATASET_PATH, DATASET_NAME)
    print(f"File saved at {p}")
