# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def mav(x):
    """Mean absolute value (MAV)"""
    return np.mean(np.abs(x), axis=2)

def std(x):
    """Standard Deviation (STD)"""
    return np.std(x, axis=2)

def var(x):
    """Variance"""
    return np.var(x, axis=2)

def maxav(x):
    """Maximum absolute Value (MaxAV)"""
    return np.max(np.abs(x), axis=2)

def rms(x):
    """Root mean square (RMS)"""
    return np.sqrt(np.mean(x**2, axis=2))

def wl(x):
    """Waveform length (WL)"""
    return np.sum(np.abs(np.diff(x, axis=2)), axis=2)

def ssc(x):
    """Slope sign changes (SSC)"""
    return np.sum((np.diff(x, axis=2)[:-1] * np.diff(x, axis=2)[1:]) < 0, axis=2)

def zc(x):
    """Zero Crossing (ZC)"""
    return np.sum(np.diff(np.sign(x), axis=2) != 0)

def log_det(x):
    """Log detector"""
    return np.exp(1 / len(x) * np.sum(np.log(x), axis=2))

def wamp(x):
    """Willison amplitude"""
    return np.sum((x > 0.2 * np.std(x)), axis=2)

def fft_values(x):
    """Frequency domain features (FFT-based) - Value"""
    return np.fft.fft(x, axis=2)

def fft_magnitude(x):
    """Frequency domain features (FFT-based) - Magntiude"""
    return np.abs(fft_values(x))

def fft_power(x):
    """Frequency domain features (FFT-based) - Power"""
    return np.square(fft_magnitude(x))

def freqs(x, srate: float = 1000.0):
    """
    Frequency domain features (FFT-based) - Frequency
    Assuming a sampling rate of 1000 Hz
    """
    return np.fft.fftfreq(x.shape[1], d=1/srate)

def total_power(x):
    """Total power"""
    return np.sum(fft_power(x), axis=2)

def mean_freq(x):
    """Mean frequency"""
    return np.sum(freqs(x) * fft_power(x), axis=2) / np.sum(fft_power(x), axis=2)

def median_freq(x):
    """Median frequency"""
    return np.median(freqs(x) * fft_power(x), axis=2)

def peak_freq(x):
    """Peak frequency"""
    return freqs(x)[np.argmax(fft_power(x), axis=2)]


FEATURE_FUNCTIONS: list = [
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
    fft_values,
    fft_magnitude,
    fft_power,
    freqs,
    total_power,
    mean_freq,
    median_freq,
    peak_freq,
]
