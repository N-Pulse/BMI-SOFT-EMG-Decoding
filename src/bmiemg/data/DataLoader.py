# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DataLoader:
    data_dir: Path
    subject: str | None = None
    session: str | None = None
    task: str | None = None
    run: str | None = None
    file_type: str = ".xdf"
    modality: str = "emg"

"""
def __post_init__(self):
    required_entries = [
        self.subject,
        self.session,
        self.modality,
    ]

    if any(x is None for x in required_entries):
        raise ValueError("subject, session, and modality must not be None")
"""
