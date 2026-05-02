# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne
import pyxdf

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pathlib import Path

from bmiemg.data.maker_archive import BIDSLoader



# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data" / "bids"

FILE: Path = DATA / "sub-05/ses-02/sourcedata/sub-05_ses-02_task-Up_run-01_raw.xdf"



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    streams, header = pyxdf.load_xdf(FILE)

    for i, stream in enumerate(streams):
        keys = stream.keys()
        info = stream["info"]
        name = info["name"][0]
        stream_type = info["type"][0]
        sfreq = info["nominal_srate"][0]
        shape = getattr(stream["time_series"], "shape", None)

        footer = stream["footer"]

        print(f"Stream {i}")
        print(f"  keys:  {keys}")
        print("  info keys")
        print(f"    name:  {name}")
        print(f"    type:  {stream_type}")
        print(f"    sfreq: {sfreq}")
        print(f"    shape: {shape}")
        print("  footer keys")
        print(f"  footer:  {footer.keys()}")
        print()

    """
    signal_stream = streams[1]
    channels = signal_stream["info"]["desc"][0]["channels"][0]["channel"]
    for i, ch in enumerate(channels):
        label = ch.get("label", [""])[0]
        ch_type = ch.get("type", [""])[0]
        unit = ch.get("unit", [""])[0]

        print(f"{i:02d} | label={label:20s} | type={ch_type:10s} | unit={unit}")


    marker_stream = streams[0]
    signal_stream = streams[1]

    data = np.asarray(signal_stream["time_series"])
    sfreq = float(signal_stream["info"]["nominal_srate"][0])

    # AUX7-AUX12 are probably your EMG channels
    emg_indices = [64, 65, 66, 67, 68, 69]
    emg_names = ["AUX7", "AUX8", "AUX9", "AUX10", "AUX11", "AUX12"]

    # XDF is samples x channels
    # MNE wants channels x samples
    # Your data is in uV, MNE expects V for EMG, so multiply by 1e-6
    emg_data = data[:, emg_indices].T * 1e-6

    info = mne.create_info(
        ch_names=emg_names,
        sfreq=sfreq,
        ch_types=["emg"] * len(emg_names),
    )

    raw_emg = mne.io.RawArray(emg_data, info)

    print(raw_emg)
    raw_emg.plot()

    marker_times = marker_stream["time_stamps"]
    marker_values = marker_stream["time_series"]

    signal_start_time = signal_stream["time_stamps"][0]

    onsets = marker_times - signal_start_time
    durations = [0.0] * len(onsets)
    descriptions = [str(marker[0]) for marker in marker_values]

    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
    )

    raw_emg.set_annotations(annotations)

    events, event_id = mne.events_from_annotations(raw_emg)

    print(event_id)
    print(events.shape)


    print("\n\n\n")
    for i in range(1, 8):
        epochs = mne.Epochs(
            raw_emg,
            events,
            event_id={"movement": i},
            baseline=None,
            preload=True,
        )

        X = epochs.get_data()
        print(X.shape)

    print("\n\n\n")
    marker_times = marker_stream["time_stamps"]
    marker_values = marker_stream["time_series"]

    signal_start_time = signal_stream["time_stamps"][0]

    marker_table = pd.DataFrame({
        "marker": [str(v[0]) for v in marker_values],
        "xdf_time": marker_times,
        "time_from_signal_start_s": marker_times - signal_start_time,
    })

    marker_table["dt_from_previous_s"] = marker_table["time_from_signal_start_s"].diff()

    print(marker_table)

    movement_markers = [
        marker
        for marker in event_id.keys()
        if str(marker).startswith("5")
    ]

    print(movement_markers)

    movement_event_id = {
        str(marker): event_id[marker]
        for marker in movement_markers
    }

    epochs_movement = mne.Epochs(
        raw_emg,
        events,
        event_id=movement_event_id,
        tmin=-0.2,
        tmax=0.5,
        baseline=None,
        preload=True,
    )

    X_movement = epochs_movement.get_data()

    print(X_movement.shape)
    """
    """
    print(emg.ch_names)

    events = mne.find_events(emg, stim_channel="TRIGGER")

    print(events)
    print(type(events))
    print(events.shape)

    # 2. Define the trigger you care about
    event_id = {
        "target": 5,   # trigger code of interest
    }

    # 3. Epoch only around that trigger
    epochs = mne.Epochs(
        emg,
        events,

        tmin=-0.2,
        tmax=0.8,
        baseline=(None, 0),
        preload=True,
    )

    print(epochs)
    print(type(epochs))
    print(epochs.event_id)
    """

    plt.show()
