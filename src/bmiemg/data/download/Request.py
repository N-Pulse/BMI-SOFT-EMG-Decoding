# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Request:
    url: str
    filename: str | None
    out_path: Path
    extra: dict = field(default_factory=dict)
