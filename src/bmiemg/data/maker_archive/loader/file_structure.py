# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne_bids

from pathlib import Path
from mne_bids import BIDSPath

from .patterns import SUBJECT_PATTERN_01, DATE_PATTERN_01



# ================================================================
# 1. Section: Checker Functions
# ================================================================
def is_dataset_bids(data_dir: Path) -> bool:
    # A. Checks if the dataset is not empty
    subfolders = [p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(subfolders) == 0:
        raise FileNotFoundError(f"There was no foulders inside {data_dir}")

    # 1. Gets the folders names inside it as list
    endings = [p.name for p in subfolders]
    subject_finds = any(SUBJECT_PATTERN_01.fullmatch(part) for part in endings)
    date_finds = any(DATE_PATTERN_01.fullmatch(part) for part in endings)

    # 2. Applies the condition check accordingly
    if subject_finds and date_finds:
        raise NameError(
            "Found both instances of subject folders and date folders"
            f"at {endings}"
        )
    elif not subject_finds and not date_finds:
        raise NameError(
            "Found neither instances of subject folders and date folders"
            f"at {endings}"
        )
    elif subject_finds:
        return True
    else:
        return False



# ================================================================
# 0. Section: Extractors
# ================================================================
def extract_perfect_bids(data_dir: Path, modality: str, file_type: str) -> list[BIDSPath]:
    return mne_bids.find_matching_paths(
            data_dir,
            datatypes=modality,
            extensions=file_type
    )

def extract_date_bids(data_dir: Path, modality: str, file_type: str) -> list[BIDSPath]:
    subfolders = [p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    all_bids_path_list = []

    for folder in subfolders:
        root = data_dir / folder

        bids_paths = mne_bids.find_matching_paths(
            root,
            datatypes=modality,
            extensions=file_type
        )
        all_bids_path_list.extend(bids_paths)

    return [item for sublist in all_bids_path_list for item in sublist]
