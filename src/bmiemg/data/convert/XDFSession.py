# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from .SignalStream import SignalStream
from .MarkerStream import MarkerStream



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class XDFSession:
    path: Path
    signal_stream: SignalStream
    marker_stream: MarkerStream
    file_header: dict
