# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Stream:
    raw_stream: dict

    @property
    def time_series(self) -> np.ndarray:
        return self.raw_stream["time_series"]

    @property
    def time_stamps(self) -> np.ndarray:
        return self.raw_stream["time_stamps"]

    @property
    def info(self) -> dict:
        return self.raw_stream["info"]

    @property
    def footer(self) -> dict:
        return self.raw_stream["footer"]

    @property
    def channels(self) -> dict:
        return self.info["desc"][0]["channels"][0]["channels"]

    @property
    def channel_names(self) -> np.ndarray:
        return np.array([ch["name"] for ch in self.channels])

    @property
    def channel_types(self) -> np.ndarray:
        return np.array([ch["type"] for ch in self.channels])

    @property
    def sfreq(self) -> float:
        return self.info["nominal_srate"][0]
