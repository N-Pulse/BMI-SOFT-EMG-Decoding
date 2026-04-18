# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from bmiemg.data.maker_archive import BIDSLoader
from bmiemg.plots import plot_emg_with_events
from bmiemg.data.features import get_emg_channel_names, get_event_labels

from pathlib import Path



# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data" / "bids"



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    bids = BIDSLoader(
        data_dir=DATA,
        modality="emg",
        file_type=".fif"
    )

    emg = bids.load_dataset(1)

    print("raw.info")
    print(emg.info)
    print()
    print("raw.ch_names")
    print(emg.ch_names)
    print()
    print("raw.annotations")
    print(emg.annotations)
    print()
