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
    emg_ch_names = [
        "AUX7",
        "AUX8",
        "AUX9",
        "AUX10",
        "AUX11",
        "AUX12",
    ],
    eeg_ch_names = [
        "FP1",
        "FPZ",
        "FP2",
        "F7",
        "F3",
        "FZ",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "M1",
        "T7",
        "C3",
        "CZ",
        "C4",
        "T8",
        "M2",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "P7",
        "P3",
        "PZ",
        "P4",
        "P8",
        "POZ",
        "O1",
        "O2",
        "EOG",
        "AF7",
        "AF3",
        "AF4",
        "AF8",
        "F5",
        "F1",
        "F2",
        "F6",
        "FC3",
        "FCZ",
        "FC4",
        "C5",
        "C1",
        "C2",
        "C6",
        "CP3",
        "CP4",
        "P5",
        "P1",
        "P2",
        "P6",
        "PO5",
        "PO3",
        "PO4",
        "PO6",
        "FT7",
        "FT8",
        "TP7",
        "TP8",
        "PO7",
        "PO8",
        "OZ"
    ],
)
