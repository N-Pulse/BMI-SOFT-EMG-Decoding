# ================================================================
# 0. Section: IMPORTS
# ================================================================
# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from pathlib import Path

from bmiemg.data.convert import session_load, ChannelSplitter



# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data" / "bids"

FILE: Path = DATA / "sub-05/ses-02/sourcedata/sub-05_ses-02_task-Up_run-01_raw.xdf"



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    session = session_load(FILE)
    biotech_splitter = ChannelSplitter()

    bio_signal = biotech_splitter.split(session)
    signal = bio_signal.attach_annotations()

    signal["EMG"].plot()

    plt.show()
