# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from ...domain import DownloadSpec


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class ZenodoSpecs(DownloadSpec):
    source: ClassVar[str] = "zenodo"

    record_id: int
    filename: str
    extract: bool = True
    extract_to: str | None = None
