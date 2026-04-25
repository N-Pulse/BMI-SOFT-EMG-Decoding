# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure



# ================================================================
# 1. Section: Functions
# ================================================================
def plot_emg(data: np.ndarray, times: np.ndarray, channels_names: np.ndarray) -> tuple[Figure, Axes]:
    peak = np.max(np.abs(data))
    offset = 1.2 * peak if peak > 0 else 1.0

    fig, ax = plt.subplots(figsize=(14, 7))
    for i in range(len(channels_names)):
        ax.plot(times, data[i] + i * offset, linewidth=1)

    # 6. Adds the emg and plot labels
    ax.set_yticks([i * offset for i in range(len(channels_names))])
    ax.set_yticklabels(channels_names)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title("EMG with events")

    plt.tight_layout()
    return fig, ax
