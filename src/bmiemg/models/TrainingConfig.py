# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class TrainingConfig:
    n_splits: int
    scoring: str
    shuffle: bool
    random_state: int = 42
