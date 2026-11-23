# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne.io as mne
import numpy as np

from matplotlib import pyplot as plt

from pathlib import Path



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_PATH: Path = Path(
    "data/pruned/raw_archive_v1/client/raw_archive_v1_train_2026-10-06_20-07-37_raw.fif"
)
#[101. 102. 104. 105. 302. 503. 504. 505. 700.]
LABELS = {101: 'rest', 102: 'fist', 104: 'open', 105: 'pinch', 302: 'left', 503: 'right', 504: "", 505: "", 700: ""}  # adjust to your codes

# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
from matplotlib.animation import FuncAnimation




WINDOW = 0.5      # seconds visible on screen
STEP = 0.02       # seconds advanced per frame
INTERVAL_MS = 20  # ms between frames -> STEP*1000 == INTERVAL_MS means real time

if __name__ == '__main__':
    data = mne.read_raw_fif(DATA_PATH)
    cropped = data.copy().crop(tmax=30).load_data()

    picks = ['AUX7', 'AUX8', 'AUX9', 'AUX10', 'AUX11', 'AUX12']
    Y, t = cropped.get_data(picks=picks, return_times=True)
    stim = cropped.get_data(picks='STI 014')[0]
    print(np.unique(stim))

    fig, axes = plt.subplots(len(picks) + 1, 1, sharex=True, figsize=(10, 9),
                             gridspec_kw={'height_ratios': [1]*len(picks) + [0.6]})
    for ax, ch, y in zip(axes, picks, Y):
        ax.plot(t, y, lw=0.8)
        ax.set_ylabel(ch, rotation=0, labelpad=25)

    ax = axes[-1]
    ax.step(t, stim, where='post', color='k', lw=1)
    ax.set_ylabel('Movement', rotation=0, labelpad=30)
    codes = np.unique(stim).astype(int)
    ax.set_yticks(codes)
    ax.set_yticklabels([LABELS.get(c, str(c)) for c in codes])
    ax.set_xlabel('Time (s)')
    fig.tight_layout()

    n_frames = int((t[-1] - WINDOW) / STEP)

    def update(i):
        start = i * STEP
        axes[-1].set_xlim(start, start + WINDOW)
        fig.suptitle(f'{start:.2f} – {start + WINDOW:.2f} s')
        return axes

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=INTERVAL_MS, blit=False, repeat=True)
    plt.show()
