# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..XDFFile import XDFFile
from .EEGStream import EEGStream
from .EMGStream import EMGStream
from .MarkerStream import MarkerStream
from .StreamConfig import StreamConfig
from .Streams import Streams
from ..utils import find_stream, split_stream



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class StreamFactory:
    config: StreamConfig

    def extract(self, file: XDFFile) -> Streams:
        streams = file.streams

        marker_stream = find_stream(streams, self.config.marker_stream_type)
        if self.config.eeg_stream_type == self.config.emg_stream_type:
            signal_stream = find_stream(streams, self.config.eeg_stream_type)

            # Separate the streams
            eeg_stream, emg_stream = split_stream(signal_stream, self.config.channel_names)
        else:
            eeg_stream = find_stream(streams, self.config.eeg_stream_type)
            emg_stream = find_stream(streams, self.config.emg_stream_type)

        eeg = EEGStream(
            time_series = eeg_stream["time_series"],
            time_stamps = eeg_stream["time_stamps"],
            info = eeg_stream["info"],
            footer=eeg_stream["footer"]
        )
        emg = EMGStream(
            time_series = emg_stream["time_series"],
            time_stamps = emg_stream["time_stamps"],
            info = emg_stream["info"],
            footer=emg_stream["footer"]
        )

        marker = MarkerStream(
            time_series = marker_stream["time_series"],
            time_stamps = marker_stream["time_stamps"],
            info = marker_stream["info"],
            footer=marker_stream["footer"]
        )

        return Streams(eeg, emg, marker)
