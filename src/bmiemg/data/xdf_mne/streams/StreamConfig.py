# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class StreamConfig:
    eeg_stream_type: str        # something like "EEG"
    emg_stream_type: str        # something like "EEG, may need spliting"
    marker_stream_type: str     # something like "Marker"
    marker_labels: dict
    data_unit: str              # uV
    default_channel_type: str
    channel_names: dict = {
        "EEG": "EEG",
        "EMG": "AUX"
    }
