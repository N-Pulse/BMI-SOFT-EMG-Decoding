# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class DownloadSpec:
    source: ClassVar[str]
    dest: str
