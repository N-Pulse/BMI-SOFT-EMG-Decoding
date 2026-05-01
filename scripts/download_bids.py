# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

from dotenv import load_dotenv
from pathlib import Path

from bmiemg.data.download.strategy import ArchiveDownloadStrategy
from bmiemg.data.download import Credentials, Request, Downloader



# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".venv")



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    credentials = Credentials(
        username=str(os.getenv("MAKER_USERNAME")),
        password=str(os.getenv("MAKER_APP_PASSWORD")),
        key=str(os.getenv("MAKER_DAV_USER"))
    )

    request = Request(
        url="EPFL N-pulse/Quality Management System/BMI/bids/",
        filename=None,
        out_path=Path("data/bids/")
    )

    archive_downloader = Downloader(
        strategy = ArchiveDownloadStrategy(),
        credentials = credentials,
    )

    response = archive_downloader.run(request)
