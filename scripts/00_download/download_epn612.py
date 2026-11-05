# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

from bmiemg.download import Downloader, ZenodoSpecs

# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

RECORD_ID = 4421500
FILENAME = "EMG-EPN612 Dataset.zip"


# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    specs = ZenodoSpecs(
        dest=str(RAW),
        record_id=RECORD_ID,
        filename=FILENAME,
        extract_to=str(RAW / "emg_epn_612"),
    )

    downloader = Downloader()
    downloader.run(specs)
