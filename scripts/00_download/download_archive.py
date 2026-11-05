# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

from pathlib import Path
from dotenv import load_dotenv

from bmiemg.download import ArchiveSpecs, Downloader

# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    specs = ArchiveSpecs(
        dest="data/bids/",
        url="EPFL N-pulse/Quality Management System/BMI/bids/",
        username=str(os.getenv("MAKER_USERNAME")),
        password=str(os.getenv("MAKER_APP_PASSWORD")),
        dav_user=str(os.getenv("MAKER_DAV_USER")),
        extra={"ignore": ["eeg", "emg"]},
    )

    downloader = Downloader()
    downloader.run(specs)
