# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Time-domain features
# ================================================================
def mav(x: np.ndarray) -> np.ndarray:
    """Mean absolute value."""
    return np.mean(np.abs(x), axis=2)

def std(x: np.ndarray) -> np.ndarray:
    """Standard deviation."""
    return np.std(x, axis=2)

def var(x: np.ndarray) -> np.ndarray:
    """Variance."""
    return np.var(x, axis=2)

def maxav(x: np.ndarray) -> np.ndarray:
    """Maximum absolute value."""
    return np.max(np.abs(x), axis=2)

def rms(x: np.ndarray) -> np.ndarray:
    """Root mean square."""
    return np.sqrt(np.mean(x**2, axis=2))

def wl(x: np.ndarray) -> np.ndarray:
    """Waveform length."""
    return np.sum(np.abs(np.diff(x, axis=2)), axis=2)

def ssc(x: np.ndarray) -> np.ndarray:
    """Slope sign changes."""
    dx = np.diff(x, axis=2)
    return np.sum((dx[:, :, :-1] * dx[:, :, 1:]) < 0, axis=2)

def zc(x: np.ndarray) -> np.ndarray:
    """Zero crossings."""
    return np.sum(np.diff(np.signbit(x), axis=2), axis=2)

def log_det(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Log detector.

    Uses abs(x) because EMG can be negative.
    """
    return np.exp(np.mean(np.log(np.abs(x) + eps), axis=2))

def wamp(x: np.ndarray, threshold: float = 20e-6) -> np.ndarray:
    """
    Willison amplitude.

    Counts how often abs(diff) exceeds a threshold.
    Threshold assumes signal is in volts.
    """
    dx = np.abs(np.diff(x, axis=2))
    return np.sum(dx > threshold, axis=2)



# ================================================================
# 2. Section: Frequency-domain scalar features
# ================================================================
def fft_power(x: np.ndarray) -> np.ndarray:
    """
    One-sided FFT power spectrum.

    Returns:
        shape (n_epochs, n_channels, n_freqs)
    """
    fft = np.fft.rfft(x, axis=2)
    return np.abs(fft) ** 2

def fft_freqs(x: np.ndarray, sfreq: float) -> np.ndarray:
    """
    One-sided FFT frequency vector.

    Returns:
        shape (n_freqs,)
    """
    return np.fft.rfftfreq(x.shape[2], d=1.0 / sfreq)

def total_power(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Total spectral power."""
    power = fft_power(x)
    return np.sum(power, axis=2)

def mean_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Mean frequency."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    denom = np.sum(power, axis=2)
    denom = np.maximum(denom, 1e-12)

    return np.sum(power * f[None, None, :], axis=2) / denom

def median_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Median frequency based on cumulative spectral power."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    cumulative_power = np.cumsum(power, axis=2)
    half_power = cumulative_power[:, :, -1:] / 2.0

    idx = np.argmax(cumulative_power >= half_power, axis=2)

    return f[idx]

def peak_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Peak frequency."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    idx = np.argmax(power, axis=2)

    return f[idx]


# ================================================================
# 3. Section: Mapped
# ================================================================
TIME_FEATURE_FUNCTIONS = [
    mav,
    std,
    var,
    maxav,
    rms,
    wl,
    ssc,
    zc,
    log_det,
    wamp,
]

FREQ_FEATURE_FUNCTIONS = [
    total_power,
    mean_freq,
    median_freq,
    peak_freq,
]
