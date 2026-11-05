"""Lazy conversion of EMG-EPN-612 signals to the deployment ADC domain.

The source Myo armband stores signed, unitless 8-bit samples at 200 Hz.  The
deployment representation used here is a centred signed view of the
STM32WBA65RI's 12-bit ADC sampled at 2000 Hz.  The transform intentionally
does not claim to recover physical volts or information above the source
Nyquist frequency.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, iirnotch, lfilter, resample_poly, sosfilt

if TYPE_CHECKING:
    from bmiemg.data.epn612.loader import EMGTrial

FloatArray: TypeAlias = NDArray[np.float32]
IntegerArray: TypeAlias = NDArray[np.signedinteger]


@dataclass(frozen=True, slots=True)
class BoardAdaptationConfig:
    """Parameters defining the source and target acquisition domains."""

    source_sampling_rate_hz: float = 200.0
    target_sampling_rate_hz: float = 2000.0
    source_adc_bits: int = 8
    target_adc_bits: int = 12
    highpass_hz: float = 20.0
    lowpass_hz: float = 500.0
    filter_order: int = 1
    notch: bool = False
    notch_hz: float = 50.0 
    

    def __post_init__(self) -> None:
        if self.source_sampling_rate_hz <= 0 or self.target_sampling_rate_hz <= 0:
            raise ValueError("Sampling rates must be positive")
        if self.source_adc_bits < 2 or not 2 <= self.target_adc_bits <= 32:
            raise ValueError("ADC resolutions must be between 2 and 32 bits")
        if not 0 < self.highpass_hz < self.lowpass_hz:
            raise ValueError("Expected 0 < high-pass cutoff < low-pass cutoff")
        if self.lowpass_hz >= self.target_sampling_rate_hz / 2:
            raise ValueError("Low-pass cutoff must be below the target Nyquist rate")
        if self.filter_order <= 0:
            raise ValueError("Filter order must be positive")
        if self.filter_order <= 0:
            raise ValueError("Filter order must be positive")
        if self.notch and not 0 < self.notch_hz < self.target_sampling_rate_hz / 2:          
            raise ValueError("notch_hz must be between 0 and the target Nyquist rate")       
    @property
    def resampling_ratio(self) -> tuple[int, int]:
        """Return the integral polyphase up/down sampling factors."""

        ratio = Fraction(
            self.target_sampling_rate_hz / self.source_sampling_rate_hz
        ).limit_denominator(100_000)
        return ratio.numerator, ratio.denominator

    @property
    def target_count_limits(self) -> tuple[int, int]:
        """Return inclusive limits of the target signed ADC representation."""

        half_range = 1 << (self.target_adc_bits - 1)
        return -half_range, half_range - 1


@dataclass(frozen=True, slots=True)
class BoardAdaptedTrial:
    """One trial represented as target-board signed ADC counts."""

    emg_counts: IntegerArray
    sampling_rate_hz: float
    adc_resolution_bits: int
    user_id: str
    trial_id: str
    gesture: str | None
    source_path: Path
    source_sampling_rate_hz: float
    source_gesture_start: int | None
    gesture_start: int | None


def normalize_signed_adc(signal: np.ndarray, resolution_bits: int) -> FloatArray:
    """Map signed ADC counts to a zero-centred fractional full-scale value."""

    if resolution_bits < 2:
        raise ValueError("ADC resolution must be at least two bits")
    array = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Signal contains NaN or infinite values")
    lower = -(1 << (resolution_bits - 1))
    upper = (1 << (resolution_bits - 1)) - 1
    if array.size and (float(array.min()) < lower or float(array.max()) > upper):
        raise ValueError(
            f"Signal exceeds signed {resolution_bits}-bit limits [{lower}, {upper}]"
        )
    return np.asarray(array / float(1 << (resolution_bits - 1)), dtype=np.float32)


def resample_emg(signal: np.ndarray, config: BoardAdaptationConfig) -> FloatArray:
    """Polyphase-resample channels along the last axis to the board rate."""

    array = np.asarray(signal, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError("Signal must have a non-empty sample axis")
    up, down = config.resampling_ratio
    result = resample_poly(array, up, down, axis=-1, padtype="line")
    return np.asarray(result, dtype=np.float32)


def mimic_board_analog_filter(
    signal: np.ndarray, config: BoardAdaptationConfig
) -> FloatArray:
    """Apply causal first-order high- and low-pass stages at 20 and 500 Hz.

    The two cascaded digital Butterworth sections approximate the board's two
    first-order analogue RC stages while retaining a causal transform suitable
    for deployment.  Processing a complete trial avoids resetting state at
    every model window.
    """

    array = np.asarray(signal, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError("Signal must have a non-empty sample axis")
    highpass = butter(
        config.filter_order,
        config.highpass_hz,
        btype="highpass",
        fs=config.target_sampling_rate_hz,
        output="sos",
    )
    lowpass = butter(
        config.filter_order,
        config.lowpass_hz,
        btype="lowpass",
        fs=config.target_sampling_rate_hz,
        output="sos",
    )
    filtered = sosfilt(highpass, array, axis=-1)
    filtered = sosfilt(lowpass, filtered, axis=-1)
    
    if config.notch:                                                                  # optional: remove mains noise
        b, a = iirnotch(config.notch_hz, Q=30.0, fs=config.target_sampling_rate_hz)   # narrow notch at notch_hz
        filtered = lfilter(b, a, filtered, axis=-1)       
        
    return np.asarray(filtered, dtype=np.float32)

def quantize_signed_adc(signal: np.ndarray, resolution_bits: int) -> IntegerArray:
    """Quantize fractional full scale to centred signed ADC counts."""

    if not 2 <= resolution_bits <= 32:
        raise ValueError("ADC resolution must be between 2 and 32 bits")
    array = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Signal contains NaN or infinite values")
    half_range = 1 << (resolution_bits - 1)
    lower, upper = -half_range, half_range - 1
    counts = np.clip(np.rint(array * half_range), lower, upper)
    dtype = np.int16 if resolution_bits <= 16 else np.int32
    return np.asarray(counts, dtype=dtype)


def board_counts_to_normalized(counts: np.ndarray, resolution_bits: int) -> FloatArray:
    """Convert centred target ADC counts back to fractional full scale."""

    return normalize_signed_adc(counts, resolution_bits)


def adapt_emg_to_board(
    signal: np.ndarray, config: BoardAdaptationConfig = BoardAdaptationConfig()
) -> IntegerArray:
    """Scale, resample, filter and quantize one EMG signal or signal batch."""

    normalized = normalize_signed_adc(signal, config.source_adc_bits)
    resampled = resample_emg(normalized, config)
    filtered = mimic_board_analog_filter(resampled, config)
    return quantize_signed_adc(filtered, config.target_adc_bits)


def adapt_trial(
    trial: EMGTrial, config: BoardAdaptationConfig = BoardAdaptationConfig()
) -> BoardAdaptedTrial:
    """Transform one loaded source trial and scale its sample annotations."""

    if not np.isclose(trial.sampling_rate_hz, config.source_sampling_rate_hz):
        raise ValueError(
            f"{trial.user_id}:{trial.trial_id} is sampled at "
            f"{trial.sampling_rate_hz} Hz, expected "
            f"{config.source_sampling_rate_hz} Hz"
        )
    up, down = config.resampling_ratio
    gesture_start = (
        None if trial.gesture_start is None else round(trial.gesture_start * up / down)
    )
    return BoardAdaptedTrial(
        emg_counts=adapt_emg_to_board(trial.emg, config),
        sampling_rate_hz=config.target_sampling_rate_hz,
        adc_resolution_bits=config.target_adc_bits,
        user_id=trial.user_id,
        trial_id=trial.trial_id,
        gesture=trial.gesture,
        source_path=trial.source_path,
        source_sampling_rate_hz=trial.sampling_rate_hz,
        source_gesture_start=trial.gesture_start,
        gesture_start=gesture_start,
    )


def iter_adapted_split(
    manifest_path: str | Path,
    split: str,
    config: BoardAdaptationConfig = BoardAdaptationConfig(),
) -> Iterator[BoardAdaptedTrial]:
    """Load and adapt manifest trials one participant JSON at a time."""

    # Imported lazily to keep this low-level adaptation module independent from
    # the public bmiemg.data.epn612 re-export module during package startup.
    from bmiemg.data.epn612.loader import load_user_trials

    manifest = Path(manifest_path)
    selections: dict[tuple[Path, str], set[str]] = defaultdict(set)
    with manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == split:
                selections[(Path(row["source_path"]), row["source_section"])].add(
                    row["trial_id"]
                )
    if not selections:
        raise ValueError(f"Split {split!r} is absent or empty in {manifest}")

    for (source_path, section), trial_ids in sorted(
        selections.items(), key=lambda item: str(item[0])
    ):
        found: set[str] = set()
        for trial in load_user_trials(source_path, section=section):  # type: ignore[arg-type]
            if trial.trial_id in trial_ids:
                found.add(trial.trial_id)
                yield adapt_trial(trial, config)
        missing = trial_ids - found
        if missing:
            raise ValueError(
                f"Manifest references missing trials in {source_path}: "
                f"{sorted(missing)}"
            )
