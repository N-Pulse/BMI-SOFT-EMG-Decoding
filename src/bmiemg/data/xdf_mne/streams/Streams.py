# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .EEGStream import EEGStream
from .EMGStream import EMGStream
from .MarkerStream import MarkerStream



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Streams:
    eeg_stream: EEGStream
    emg_stream: EMGStream
    marker_stream: MarkerStream
