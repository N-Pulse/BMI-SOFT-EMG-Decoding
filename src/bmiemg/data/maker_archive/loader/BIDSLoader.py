# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from mne_bids import BIDSPath

from ...DataLoader import DataLoader
from .file_structure import (
    is_dataset_bids,
    extract_date_bids,
    extract_perfect_bids
)



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class BIDSLoader(DataLoader):
    def is_dataset_bids(self) -> bool:
        return is_dataset_bids(self.data_dir)


    @property
    def bids_paths(self) -> list[BIDSPath]:
        is_perfect = self.is_dataset_bids()

        if is_perfect:
            bids_paths = extract_perfect_bids(self.data_dir, self.modality, self.file_type)
        else:
            bids_paths = extract_date_bids(self.data_dir, self.modality, self.file_type)

        return bids_paths
