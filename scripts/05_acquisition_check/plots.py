# ================================================================
# 0. Section: IMPORTS
# ================================================================
import matplotlib
import numpy as np

from pathlib import Path
from collections import Counter

from loaders import Recording

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ================================================================
# 1. Section: MARKER PLOT
# ================================================================
def plot_markers(
    rec: Recording,
    channels: list[int],
    window_s: tuple[float, float],
    max_codes: int,
    out_path: Path,
) -> None:
    """Overview of marker times over the EMG envelope, then one panel per
    marker code with the raw channels around its first occurrence."""
    # 1. Codes to plot, most frequent first
    counts = Counter(code for _, code in rec.markers)
    codes = [c for c, _ in counts.most_common(max_codes)]
    colors = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(codes)}
    pre, post = int(-window_s[0] * rec.fs), int(window_s[1] * rec.fs)
    first = {c: next(i for i, m in rec.markers if m == c) for c in codes}

    # 2. Read each channel once: 100 ms RMS envelope and the snippets
    bin_len = max(int(0.1 * rec.fs), 1)
    envelopes, snippets = {}, {c: {} for c in codes}
    for ch in channels:
        x = rec.get_channel(ch)
        x = np.nan_to_num(x - np.nanmedian(x))
        n_bins = x.size // bin_len
        env = np.sqrt((x[: n_bins * bin_len].reshape(n_bins, bin_len) ** 2).mean(1))
        envelopes[ch] = env / (np.percentile(env, 99) or 1)
        for c, onset in first.items():
            seg = x[max(onset - pre, 0) : onset + post]
            snippets[c][ch] = (seg - seg.mean()) / (seg.std() or 1)
        del x

    # 3. Layout: two overview rows, then a grid of per-code panels
    n_cols = 3
    n_rows = int(np.ceil(len(codes) / n_cols))
    fig = plt.figure(figsize=(16, 6 + 3.2 * n_rows))
    grid = fig.add_gridspec(2 + n_rows, n_cols, height_ratios=[1, 2] + [2] * n_rows)

    # 4. Marker raster
    ax = fig.add_subplot(grid[0, :])
    for i, code in rec.markers:
        if code in colors:
            ax.axvline(i / rec.fs, color=colors[code], lw=0.8)
    ax.set_yticks([])
    ax.set_xlim(0, rec.duration_s)
    ax.set_title(f"{rec.path.name}: {rec.n_markers} markers ({rec.marker_source})")
    ax.legend(
        handles=[
            plt.Line2D([], [], color=colors[c], label=f"{c} (n={counts[c]})")
            for c in codes
        ],
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=8,
    )

    # 5. Envelope overview with markers
    ax = fig.add_subplot(grid[1, :], sharex=fig.axes[0])
    for k, ch in enumerate(channels):
        t = np.arange(envelopes[ch].size) * bin_len / rec.fs
        ax.plot(t, envelopes[ch] + k, lw=0.5, color="k")
    for i, code in rec.markers:
        if code in colors:
            ax.axvline(i / rec.fs, color=colors[code], lw=0.4, alpha=0.5)
    ax.set_yticks(range(len(channels)), [rec.names[ch] for ch in channels], fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_title("100 ms RMS envelope (normalised per channel)")

    # 6. One panel per code around its first occurrence
    for j, code in enumerate(codes):
        ax = fig.add_subplot(grid[2 + j // n_cols, j % n_cols])
        onset = first[code]
        for k, ch in enumerate(channels):
            seg = snippets[code][ch]
            t = (np.arange(seg.size) + max(onset - pre, 0) - onset) / rec.fs
            ax.plot(t, seg * 0.3 + k, lw=0.4, color="k")
        ax.axvline(0, color=colors[code], lw=1.5)
        ax.set_yticks(
            range(len(channels)), [rec.names[ch] for ch in channels], fontsize=6
        )
        ax.set_title(f"code {code}, first at {onset / rec.fs:.1f} s", fontsize=9)
        ax.set_xlabel("time from marker (s)", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
