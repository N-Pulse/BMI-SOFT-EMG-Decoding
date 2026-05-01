# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Credentials:
    username: str
    password: str
    key: str
    extra: dict = field(default_factory=dict)
