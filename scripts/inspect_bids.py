# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from bmiemg.data.maker_archive import BIDSLoader
from bmiemg.plots import plot_emg, add_events
from bmiemg.data.features import get_emg_channel_names, get_event_labels

from pathlib import Path



# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data" / "bids"



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 0. Start the BIDS and get a session
    bids = BIDSLoader(
        data_dir=DATA,
        modality="emg",
        file_type=".fif"
    )
    emg = bids.load_dataset(1)

    # 1. Extract the data for plotting
    channels_names = np.asarray(get_emg_channel_names(emg))
    data = np.asarray(emg.get_data(picks=channels_names))
    times = np.asarray(emg.times)
    visible_annotations = get_event_labels(emg, times[0], times[-1])

    # 2. Center the data (the plotting will create offsets)
    data = data - np.mean(data, axis=0)

    fig, ax = plot_emg(data, times, channels_names)
    ax = add_events(ax, visible_annotations)
    plt.show()
