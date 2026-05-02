# ================================================================
# 0. Section: IMPORTS
# ================================================================
from bmiemg.data.maker_archive import BIDSLoader

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
    print()
    print(bids.is_dataset_bids())
    print()
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
