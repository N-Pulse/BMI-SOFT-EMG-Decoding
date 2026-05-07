# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pyxdf
from pathlib import Path

from .MarkerStream import MarkerStream
from .SignalStream import SignalStream
from .XDFSession import XDFSession



# ================================================================
# 1. Section: Functions
# ================================================================
def load(path: Path) -> XDFSession:
    # 0. Loads the data
    streams, file_header = pyxdf.load_xdf(path)

    # 1. Split the main streams (signal/marker)
    signal_stream = SignalStream(streams[1])
    marker_stream = MarkerStream(streams[0], signal_stream.time_stamps[0])

    return XDFSession(path, signal_stream, marker_stream, file_header)
