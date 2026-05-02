# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pyxdf

from dataclasses import dataclass
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class XDFFile:
    file_path: Path

    @property
    def _data(self) -> tuple[list[dict], dict]:
       return pyxdf.load_xdf(self.file_path)

    @property
    def streams(self) -> list[dict]:
        return self._data[0]

    @property
    def header(self) -> dict:
        return self._data[1]
