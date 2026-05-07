# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from dataclasses import dataclass, field
from copy import deepcopy

from ..XDFSession import XDFSession
from ..streams import SignalStream
from ..BioSignalRecording import BioSignalRecording
from .ChannelMap import ChannelMap, BIOTECH_MAP



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ChannelSplitter:
    ch_map: ChannelMap = field(default_factory=lambda: BIOTECH_MAP)

    def split(self, session: XDFSession) -> BioSignalRecording:
        # 0. Extract the data
        signal_stream = session.signal_stream
        channel_names = signal_stream.channel_names

        # 1. Get the channel index where each modality lives
        eeg_idx = np.flatnonzero(np.isin(channel_names, self.ch_map.eeg_ch_names))
        emg_idx = np.flatnonzero(np.isin(channel_names, self.ch_map.emg_ch_names))

        # 1.A Validate the channel map findings
        if eeg_idx.size == 0:
            raise ValueError("No EEG channels found.")
        if emg_idx.size == 0:
            raise ValueError("No EMG channels found.")

        # 2. Split and convert to desired file format
        eeg_raw = channel_subset_raw(
            stream=signal_stream,
            indices=eeg_idx,
            stream_type="EEG",
            segment_id=2,
        )
        emg_raw = channel_subset_raw(
            stream=signal_stream,
            indices=emg_idx,
            stream_type="EMG",
            segment_id=3,
        )
        annotations = session.marker_stream.to_annotation()

        return BioSignalRecording(eeg_raw, emg_raw, annotations)




# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def channel_subset_raw(
    stream: SignalStream,
    indices: np.ndarray,
    stream_type: str,
    segment_id: int,
) -> mne.io.RawArray:
    out = deepcopy(stream)

    out.raw_stream

    out.raw_stream["info"]["type"] = [stream_type]
    out.raw_stream["info"]["channel_count"] = [str(len(indices))]
    out.raw_stream["info"]["segment_id"] = segment_id

    out.raw_stream["info"]["desc"][0]["channels"][0]["channel"] = [
        deepcopy(stream.channels[i]) for i in indices
    ]

    out.raw_stream["time_series"] = out.time_series[:, indices]
    out.mono_signal = True

    return out.to_raw()
