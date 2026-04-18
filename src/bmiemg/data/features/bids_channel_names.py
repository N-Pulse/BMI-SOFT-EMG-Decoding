# ================================================================
# 0. Section: IMPORTS
# ================================================================
from mne.io.fiff.raw import Raw



# ================================================================
# 1. Section: Functions
# ================================================================
def get_emg_channel_names(emg: Raw) -> list:
    return [
        ch for ch, ch_type in zip(emg.ch_names, emg.get_channel_types())
        if ch_type == "emg"
    ]
