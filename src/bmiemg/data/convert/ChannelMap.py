# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ChannelMap:
    emg_ch_names: list[str]
    eeg_ch_names: list[str]


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Deafult Setups
# ──────────────────────────────────────────────────────
BIOTECH_MAP: ChannelMap = ChannelMap(
    emg_ch_names = [""],
    eeg_ch_names = [""],
)
