# ================================================================
# 0. Section: IMPORTS
# ================================================================
import re
import numpy as np

from dataclasses import dataclass

# ================================================================
# 1. Section: CHANNEL CLASSIFICATION
# ================================================================
# 10-10 system plus the old 10-20 names (T3/T4/T5/T6) and mastoids/earlobes
POSITIONS = {
    label.upper()
    for label in (
        "Fp1 Fpz Fp2 AF9 AF7 AF5 AF3 AF1 AFz AF2 AF4 AF6 AF8 AF10 "
        "F9 F7 F5 F3 F1 Fz F2 F4 F6 F8 F10 FT9 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 FT10 "
        "T9 T7 C5 C3 C1 Cz C2 C4 C6 T8 T10 TP9 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 TP10 "
        "P9 P7 P5 P3 P1 Pz P2 P4 P6 P8 P10 PO9 PO7 PO5 PO3 PO1 POz PO2 PO4 PO6 PO8 PO10 "
        "O1 Oz O2 O9 O10 Iz I1 I2 T3 T4 T5 T6 M1 M2 A1 A2"
    ).split()
}
NUMBERED = re.compile(r"^[A-Za-z]*\s*\d+$")


def classify(
    name: str,
    ch_type: str,
    emg_patterns: list[str],
    eeg_patterns: list[str],
    ignore_patterns: list[str],
) -> str:
    """Return 'emg', 'eeg' or 'other' from the name first, then the type."""
    if any(re.fullmatch(p, name, re.IGNORECASE) for p in ignore_patterns):
        return "other"
    if any(re.fullmatch(p, name, re.IGNORECASE) for p in emg_patterns):
        return "emg"
    if name.upper() in POSITIONS or any(
        re.fullmatch(p, name, re.IGNORECASE) for p in eeg_patterns
    ):
        return "eeg"
    if ch_type.upper() in ("EMG", "AUX"):
        return "emg"
    if ch_type.upper() == "EEG" and not name.startswith("<unlabeled"):
        return "eeg"
    return "other"


def position_status(name: str) -> str:
    if name.upper() in POSITIONS:
        return "mapped"
    if NUMBERED.match(name):
        return "numbered"
    return "unknown"


# ================================================================
# 2. Section: SIGNAL QUALITY
# ================================================================
@dataclass
class Thresholds:
    flat_fraction: float
    clip_fraction: float
    nan_fraction: float
    line_freq: float
    line_noise_ratio: float
    robust_z: float
    emg_band: tuple[float, float]
    emg_min_band_ratio: float
    welch_max_segments: int


@dataclass
class ChannelQuality:
    name: str
    kind: str
    std: float
    nan_frac: float
    flat_frac: float
    clip_frac: float
    line_ratio: float
    emg_band_ratio: float
    robust_z: float = 0.0
    dead_reasons: tuple[str, ...] = ()
    warn_reasons: tuple[str, ...] = ()

    @property
    def dead(self) -> bool:
        return bool(self.dead_reasons)


def channel_features(
    x: np.ndarray, fs: float, name: str, kind: str, th: Thresholds
) -> ChannelQuality:
    # 1. Missing samples
    nan_mask = ~np.isfinite(x)
    nan_frac = float(nan_mask.mean()) if x.size else 1.0
    x = x[~nan_mask]
    if x.size < 2 * fs:
        return ChannelQuality(
            name,
            kind,
            0.0,
            nan_frac,
            1.0,
            0.0,
            0.0,
            0.0,
            dead_reasons=("too few valid samples",),
        )

    # 2. Amplitude features (unit-free so they work for V and uV)
    flat_frac = float(np.mean(np.diff(x) == 0))
    clip_frac = max(float(np.mean(x == x.max())), float(np.mean(x == x.min())))

    # 3. Spectral features from a light Welch estimate
    freqs, psd = welch(x, fs, th.welch_max_segments)
    total = psd[(freqs >= 1) & (freqs < fs / 2)].sum()
    line = psd[np.abs(freqs - th.line_freq) <= 1].sum()
    lo, hi = th.emg_band[0], min(th.emg_band[1], fs / 2)
    band = psd[(freqs >= lo) & (freqs <= hi)].sum()
    line_ratio = float(line / total) if total > 0 else 0.0
    band_ratio = float(band / total) if total > 0 else 0.0

    return ChannelQuality(
        name,
        kind,
        float(x.std()),
        nan_frac,
        flat_frac,
        clip_frac,
        line_ratio,
        band_ratio,
    )


def welch(x: np.ndarray, fs: float, max_segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Average periodogram of up to max_segments 2 s Hann windows spread
    evenly over the recording."""
    nper = int(2 * fs)
    starts = np.linspace(0, x.size - nper, min(max_segments, x.size // nper)).astype(
        int
    )
    window = np.hanning(nper)
    psd = np.zeros(nper // 2 + 1)
    for s in starts:
        seg = x[s : s + nper]
        psd += np.abs(np.fft.rfft((seg - seg.mean()) * window)) ** 2
    return np.fft.rfftfreq(nper, 1 / fs), psd / max(len(starts), 1)


def judge(group: list[ChannelQuality], th: Thresholds) -> None:
    """Fill dead/warn reasons in place; the RMS outlier test compares each
    channel with the others of the same kind."""
    # 1. Robust z-score of log std within the group
    alive = [q for q in group if q.std > 0]
    if len(alive) >= 3:
        logs = np.log10([q.std for q in alive])
        med = np.median(logs)
        mad = np.median(np.abs(logs - med)) * 1.4826 or 1e-12
        for q, v in zip(alive, logs):
            q.robust_z = float((v - med) / mad)

    # 2. Dead and warning reasons per channel
    for q in group:
        dead, warn = list(q.dead_reasons), []
        if q.std == 0 or q.flat_frac >= th.flat_fraction:
            dead.append(f"flat ({q.flat_frac:.0%} repeated samples)")
        if q.nan_frac >= th.nan_fraction:
            dead.append(f"NaN ({q.nan_frac:.0%})")
        elif q.clip_frac >= th.clip_fraction:
            dead.append(f"clipped ({q.clip_frac:.1%} at min/max)")
        if q.line_ratio >= th.line_noise_ratio:
            warn.append(f"line noise {q.line_ratio:.0%} of power")
        if abs(q.robust_z) >= th.robust_z:
            warn.append(f"amplitude outlier (z={q.robust_z:+.1f})")
        if q.kind == "emg" and q.emg_band_ratio < th.emg_min_band_ratio:
            warn.append(f"only {q.emg_band_ratio:.0%} of power in EMG band")
        q.dead_reasons, q.warn_reasons = tuple(dict.fromkeys(dead)), tuple(warn)
