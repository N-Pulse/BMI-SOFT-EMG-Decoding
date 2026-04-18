# ================================================================
# 0. Section: IMPORTS
# ================================================================
from bmiemg.data.maker_archive.loader.BIDSLoader import BIDSLoader
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
        subject="05",
        session="01",
        task="Side",
        run="01",
        file_type=".fif"
    )

    print()
    print(bids.bids_paths)
    print()
