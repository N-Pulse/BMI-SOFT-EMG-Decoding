# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np
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
    # 1. Load the session
    session = session_load(FILE)
    biotech_splitter = ChannelSplitter()
    bio_signal = biotech_splitter.split(session)
    signal = bio_signal.attach_annotations()
    emg_signal = signal["EMG"]

    # 2.
    movement_labels = sorted({
        desc for desc in emg_signal.annotations.description
        if str(desc).startswith("3")
    })
    print(movement_labels)


    events, event_id = mne.events_from_annotations(
        emg_signal,
        event_id='32101',
    )

    epochs = mne.Epochs(
        raw=emg_signal,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=2.0,
        baseline=None,
        preload=True,
        metadata=None,
    )


    epochs.plot()
    plt.show()
