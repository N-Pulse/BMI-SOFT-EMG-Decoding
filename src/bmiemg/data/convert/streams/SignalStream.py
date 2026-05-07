# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from dataclasses import dataclass

from .Stream import Stream



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SignalStream(Stream):
    mono_signal: bool = False

    @property
    def main_ch_type(self) -> str:
        if not self.mono_signal:
            raise ValueError(
                "To raw only works when the signal is mono (only one modality)"
            )

        unique_types = np.unique(self.channel_types)

        if len(unique_types) > 1:
            raise ValueError(
                "The channel splitting has an issue, should only be one type of "
                f"channels, instead got: {unique_types}"
            )

        if unique_types[0].lower() == "aux":
            return "EMG"

        return unique_types[0]



    def to_raw(self) -> mne.io.RawArray:
        if not self.mono_signal:
            raise ValueError(
                "To raw only works when the signal is mono (only one modality)"
            )

        data = (self.time_series * 1e-6).T

        mne_info = mne.create_info(
            ch_names = list(self.channel_names),
            sfreq = self.sfreq,
            ch_types=self.main_ch_type.lower(),
        )

        return mne.io.RawArray(data, mne_info)
