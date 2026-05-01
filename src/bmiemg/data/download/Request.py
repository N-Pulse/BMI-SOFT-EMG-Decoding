# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Request:
    url: str
    filename: str | None
    out_path: Path
