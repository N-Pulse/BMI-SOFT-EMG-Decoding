# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from datetime import datetime
from collections import Counter

from plots import plot_markers
from loaders import Recording, load_recording
from checks import (
    Thresholds,
    ChannelQuality,
    judge,
    classify,
    position_status,
    channel_features,
)


# ================================================================
# 1. Section: INPUTS
# ================================================================
FILE_PATH = Path("data/test_data/sub-05_ses-06_task-Up_run-01_raw.xdf")
OUTPUT_DIR = Path("data/acquisition_reports/")

N_EMG_CHANNELS = 6  # N, must match exactly
M_EEG_CHANNELS = 64  # M, must match exactly
P_EMG_DEAD_ALLOWED = 0  # P
Q_EEG_DEAD_ALLOWED = 6  # Q
EXPECTED_FS = None  # Hz, None skips the check

EXPECTED_MARKER_CODES = None  # e.g. ["303", "304"], None skips the check

PLOT = True
PLOT_WINDOW_S = (-1.0, 3.0)  # around the first marker of each code
PLOT_EEG_CHANNELS = ["C3", "Cz", "C4"]  # shown below the EMG channels
PLOT_MAX_CODES = 12

EMG_PATTERNS = [r"AUX\d+", r"EMG\d+"]
EEG_PATTERNS = [r"EEG\d+"]
IGNORE_PATTERNS = [r"TRIGGER", r"EOG\d*", r"MISC\d+", r"STATUS"]

THRESHOLDS = Thresholds(
    flat_fraction=0.95,
    clip_fraction=0.01,
    nan_fraction=0.05,
    line_freq=50.0,
    line_noise_ratio=0.5,
    robust_z=5.0,
    emg_band=(20.0, 450.0),
    emg_min_band_ratio=0.5,
    welch_max_segments=200,
)


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def mark(ok: bool, warn_only: bool = False) -> str:
    return "[PASS]" if ok else ("[WARN]" if warn_only else "[FAIL]")


def build_report(
    rec: Recording, kinds: list[str], quality: dict[str, list[ChannelQuality]]
) -> tuple[str, int]:
    lines: list[str] = []
    fails = 0

    def add(ok: bool, text: str, warn_only: bool = False) -> None:
        nonlocal fails
        fails += (not ok) and not warn_only
        lines.append(f"{mark(ok, warn_only)} {text}")

    emg = [n for n, k in zip(rec.names, kinds) if k == "emg"]
    eeg = [n for n, k in zip(rec.names, kinds) if k == "eeg"]
    other = [n for n, k in zip(rec.names, kinds) if k == "other"]

    # 1. Recording overview
    lines += [
        "Acquisition check",
        f"File:       {rec.path}",
        f"Checked:    {datetime.now():%Y-%m-%d %H:%M}",
        f"Rate:       {rec.fs:g} Hz",
        f"Duration:   {rec.duration_s / 60:.1f} min ({rec.n_samples} samples)",
        f"Channels:   {len(rec.names)} total, {len(emg)} EMG, {len(eeg)} EEG, {len(other)} other",
        f"Markers:    {rec.n_markers} ({rec.marker_source})",
        "",
        "Checks",
    ]

    # 2. Loader and recording-level checks
    for note in rec.notes:
        add(False, note)
    if EXPECTED_FS is not None:
        add(
            rec.fs == EXPECTED_FS,
            f"Sampling rate {rec.fs:g} Hz (expected {EXPECTED_FS:g} Hz)",
        )
    add(rec.n_markers > 0, f"Markers present: {rec.n_markers}")
    codes = Counter(code for _, code in rec.markers)
    if EXPECTED_MARKER_CODES is not None:
        missing = [c for c in EXPECTED_MARKER_CODES if c not in codes]
        add(not missing, f"Expected marker codes missing: {missing or 'none'}")
    duplicates = [n for n, c in Counter(rec.names).items() if c > 1]
    add(not duplicates, f"Duplicate channel names: {duplicates or 'none'}")

    # 3. Channel counts
    add(
        len(emg) == N_EMG_CHANNELS,
        f"EMG channels: {len(emg)} (expected exactly {N_EMG_CHANNELS})",
    )
    add(
        len(eeg) == M_EEG_CHANNELS,
        f"EEG channels: {len(eeg)} (expected exactly {M_EEG_CHANNELS})",
    )

    # 4. EEG positions
    status = {n: position_status(n) for n in eeg}
    numbered = [n for n, s in status.items() if s == "numbered"]
    unknown = [n for n, s in status.items() if s == "unknown"]
    add(
        not numbered,
        f"EEG numbered without a 10-20 map: {len(numbered)}"
        + (f" ({_short(numbered)})" if numbered else ""),
    )
    add(
        not unknown,
        f"EEG names not in 10-20/10-10: {len(unknown)}"
        + (f" ({_short(unknown)})" if unknown else ""),
    )

    # 5. Dead and noisy channels
    for kind, allowed in (("emg", P_EMG_DEAD_ALLOWED), ("eeg", Q_EEG_DEAD_ALLOWED)):
        dead = [q for q in quality[kind] if q.dead]
        noisy = [q for q in quality[kind] if q.warn_reasons and not q.dead]
        add(
            len(dead) <= allowed,
            f"{kind.upper()} dead channels: {len(dead)} (allowed {allowed})",
        )
        for q in dead:
            lines.append(f"         - {q.name}: {', '.join(q.dead_reasons)}")
        add(
            not noisy,
            f"{kind.upper()} suspicious channels: {len(noisy)}",
            warn_only=True,
        )
        for q in noisy:
            lines.append(f"         - {q.name}: {', '.join(q.warn_reasons)}")

    # 6. Per-channel table
    lines += [
        "",
        f"Verdict: {'OK' if fails == 0 else f'{fails} FAILED CHECK(S)'}",
        "",
        "Channel features",
    ]
    lines.append(
        f"{'name':<14}{'kind':<5}{'std':>11}{'nan':>7}{'flat':>7}{'clip':>7}{'line':>7}{'emgband':>9}{'z':>7}  status"
    )
    for kind in ("emg", "eeg"):
        for q in quality[kind]:
            state = "DEAD" if q.dead else ("WARN" if q.warn_reasons else "ok")
            lines.append(
                f"{q.name:<14}{kind:<5}{q.std:>11.3g}{q.nan_frac:>7.1%}{q.flat_frac:>7.1%}{q.clip_frac:>7.1%}"
                f"{q.line_ratio:>7.1%}{q.emg_band_ratio:>9.1%}{q.robust_z:>+7.1f}  {state}"
            )
    if other:
        lines += ["", f"Ignored channels: {_short(other, 40)}"]

    # 7. Marker codes
    if codes:
        lines += [
            "",
            "Marker codes",
            f"{'code':<16}{'count':>7}{'first (s)':>11}{'last (s)':>10}",
        ]
        for code, n in sorted(codes.items(), key=lambda kv: -kv[1]):
            times = [i / rec.fs for i, c in rec.markers if c == code]
            lines.append(f"{code:<16}{n:>7}{times[0]:>11.1f}{times[-1]:>10.1f}")
    return "\n".join(lines), fails


def _short(items: list[str], limit: int = 10) -> str:
    return ", ".join(items[:limit]) + (
        f", ... (+{len(items) - limit})" if len(items) > limit else ""
    )


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Load the recording
    rec = load_recording(FILE_PATH)

    # 2. Classify channels and compute features one channel at a time
    kinds = [
        classify(n, t, EMG_PATTERNS, EEG_PATTERNS, IGNORE_PATTERNS)
        for n, t in zip(rec.names, rec.types)
    ]
    quality: dict[str, list[ChannelQuality]] = {"emg": [], "eeg": []}
    for i, (name, kind) in enumerate(zip(rec.names, kinds)):
        if kind in quality:
            quality[kind].append(
                channel_features(rec.get_channel(i), rec.fs, name, kind, THRESHOLDS)
            )
    for group in quality.values():
        judge(group, THRESHOLDS)

    # 3. Print and save the summary
    report, _ = build_report(rec, kinds, quality)
    print(report)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{FILE_PATH.stem}_check.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {out}")

    # 4. Plot markers over the EMG and chosen EEG channels
    if PLOT and rec.markers:
        wanted = {n.upper() for n in PLOT_EEG_CHANNELS}
        channels = [i for i, k in enumerate(kinds) if k == "emg"]
        channels += [
            i
            for i, n in enumerate(rec.names)
            if kinds[i] == "eeg" and n.upper() in wanted
        ]
        png = OUTPUT_DIR / f"{FILE_PATH.stem}_markers.png"
        plot_markers(rec, channels, PLOT_WINDOW_S, PLOT_MAX_CODES, png)
        print(f"Plot saved to {png}")
