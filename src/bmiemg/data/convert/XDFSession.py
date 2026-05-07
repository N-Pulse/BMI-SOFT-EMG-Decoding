# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from .streams import SignalStream, MarkerStream



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class XDFSession:
    path: Path
    signal_stream: SignalStream
    marker_stream: MarkerStream
    file_header: dict
