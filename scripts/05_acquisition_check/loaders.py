# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pyxdf
import numpy as np

from pathlib import Path
from typing import Callable
from dataclasses import dataclass, field


# ================================================================
# 1. Section: RECORDING
# ================================================================
@dataclass
class Recording:
    path: Path
    fs: float
    names: list[str]
    types: list[str]
    n_samples: int
    markers: list[tuple[int, str]]
    get_channel: Callable[[int], np.ndarray]
    notes: list[str] = field(default_factory=list)
    marker_source: str = "file"

    @property
    def n_markers(self) -> int:
        return len(self.markers)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


def load_recording(path: Path) -> Recording:
    # 1. Pick the reader from the suffix
    suffix = path.suffix.lower()
    if suffix == ".xdf":
        rec = load_xdf(path)
    elif suffix in (".eeg", ".vhdr"):
        rec = load_brainvision(path.with_suffix(".vhdr"))
    else:
        raise ValueError(
            f"Unsupported file type {suffix!r}, allowed: .xdf, .eeg, .vhdr"
        )

    # 2. Fall back on a trigger channel when the file has no marker list:
    #    the one named TRIGGER, else the last column (where eego writes it)
    if not rec.markers:
        named = [i for i, n in enumerate(rec.names) if n.upper() == "TRIGGER"]
        last = len(rec.names) - 1
        for i in dict.fromkeys(named + [last]):
            x = np.nan_to_num(rec.get_channel(i))
            if _is_trigger(x):
                rec.markers = markers_from_trigger(x)
                rec.marker_source = f"trigger channel {rec.names[i]} (column {i + 1})"
                if i not in named and named:
                    rec.notes.append(
                        f"Channel named TRIGGER (column {named[0] + 1}) holds no codes, "
                        f"the trigger is in column {i + 1}: labels are shifted"
                    )
                break
    return rec


def _is_trigger(x: np.ndarray) -> bool:
    integer = bool(np.all(x == np.round(x)))
    return integer and 1 < np.unique(x).size <= 256 and bool(np.any(x != 0))


def markers_from_trigger(x: np.ndarray) -> list[tuple[int, str]]:
    """One marker per change of the trigger channel to a non-zero value."""
    x = np.nan_to_num(x)
    onsets = np.flatnonzero((x[1:] != x[:-1]) & (x[1:] != 0)) + 1
    if x.size and x[0] != 0:
        onsets = np.r_[0, onsets]
    return [(int(i), _code(x[i])) for i in onsets]


def _code(value) -> str:
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    except (TypeError, ValueError):
        return str(value)


# ================================================================
# 2. Section: XDF
# ================================================================
def load_xdf(path: Path) -> Recording:
    # 1. Load every stream (pyxdf cannot read a single one lazily)
    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=False)
    notes: list[str] = []

    # 2. Split marker streams (irregular rate) from signal streams
    markers = [s for s in streams if float(s["info"]["nominal_srate"][0]) == 0]
    signals = [s for s in streams if float(s["info"]["nominal_srate"][0]) > 0]
    if not signals:
        raise ValueError(f"No regular-rate signal stream in {path}")

    # 3. Keep the widest signal stream, free the rest
    signal = max(signals, key=lambda s: int(s["info"]["channel_count"][0]))
    if len(signals) > 1:
        others = [s["info"]["name"][0] for s in signals if s is not signal]
        notes.append(
            f"Several signal streams, using {signal['info']['name'][0]!r}, ignoring {others}"
        )
    info = signal["info"]
    data = np.asarray(signal["time_series"])
    stamps = np.asarray(signal["time_stamps"])
    fs = float(info["nominal_srate"][0])

    # 4. Place every marker on the signal's sample axis
    events = [
        (int(np.searchsorted(stamps, t)), _code(v[0] if np.ndim(v) else v))
        for s in markers
        for t, v in zip(s["time_stamps"], s["time_series"])
    ]
    events = sorted((min(i, len(stamps) - 1), c) for i, c in events)
    del streams, signals, markers

    # 5. Read channel labels and types from the stream description
    names, types = _xdf_channels(info)
    n_channels = data.shape[1]
    if len(names) != n_channels:
        notes.append(
            f"Header lists {len(names)} channel labels but the data has {n_channels} columns"
        )
    names = (names + [f"<unlabeled {i + 1}>" for i in range(len(names), n_channels)])[
        :n_channels
    ]
    types = (types + [""] * n_channels)[:n_channels]

    # 6. Compare nominal and effective sampling rate (dropped samples)
    if len(stamps) > 1:
        effective = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        if abs(effective - fs) / fs > 0.01:
            notes.append(
                f"Effective rate {effective:.1f} Hz differs from nominal {fs:g} Hz by more than 1%"
            )

    return Recording(
        path=path,
        fs=fs,
        names=names,
        types=types,
        n_samples=data.shape[0],
        markers=events,
        get_channel=lambda i: data[:, i].astype(np.float64),
        notes=notes,
    )


def _xdf_channels(info: dict) -> tuple[list[str], list[str]]:
    try:
        channels = info["desc"][0]["channels"][0]["channel"]
    except (KeyError, IndexError, TypeError):
        return [], []
    names = [(c.get("label") or [""])[0] or "" for c in channels]
    types = [(c.get("type") or [""])[0] or "" for c in channels]
    return names, types


# ================================================================
# 3. Section: BRAINVISION
# ================================================================
BV_FORMATS = {"IEEE_FLOAT_32": np.float32, "INT_16": np.int16, "INT_32": np.int32}


def load_brainvision(vhdr: Path) -> Recording:
    # 1. Parse the header
    header = _read_ini(vhdr)
    common, binary = header["Common Infos"], header.get("Binary Infos", {})
    n_channels = int(common["NumberOfChannels"])
    fs = 1e6 / float(common["SamplingInterval"])
    fmt = binary.get("BinaryFormat", "INT_16")
    if fmt not in BV_FORMATS:
        raise ValueError(
            f"Unsupported BinaryFormat {fmt!r}, allowed: {list(BV_FORMATS)}"
        )
    multiplexed = common.get("DataOrientation", "MULTIPLEXED") == "MULTIPLEXED"

    # 2. Channel names and resolutions
    names, scales = [], []
    for i in range(1, n_channels + 1):
        fields = header["Channel Infos"].get(f"Ch{i}", f"<unlabeled {i}>").split(",")
        names.append(fields[0].replace("\\1", ","))
        scales.append(float(fields[2]) if len(fields) > 2 and fields[2] else 1.0)

    # 3. Memory-map the binary data so only one channel is read at a time
    raw = np.memmap(vhdr.parent / common["DataFile"], dtype=BV_FORMATS[fmt], mode="r")
    n_samples = raw.size // n_channels
    data = raw[: n_samples * n_channels].reshape(
        (n_samples, n_channels) if multiplexed else (n_channels, n_samples)
    )

    def get_channel(i: int) -> np.ndarray:
        column = data[:, i] if multiplexed else data[i]
        return np.asarray(column, dtype=np.float64) * scales[i]

    # 4. Read markers (Type,Description,Position,...) from the marker file
    markers: list[tuple[int, str]] = []
    marker_file = common.get("MarkerFile")
    if marker_file and (vhdr.parent / marker_file).exists():
        entries = _read_ini(vhdr.parent / marker_file).get("Marker Infos", {})
        for entry in entries.values():
            fields = entry.split(",")
            if len(fields) > 2 and fields[2].strip().isdigit():
                markers.append((int(fields[2]) - 1, fields[1] or fields[0]))

    return Recording(
        path=vhdr,
        fs=fs,
        names=names,
        types=[""] * n_channels,
        n_samples=n_samples,
        markers=sorted(markers),
        get_channel=get_channel,
    )


def _read_ini(path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
        elif "=" in line:
            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()
    return sections
