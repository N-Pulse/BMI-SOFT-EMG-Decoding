import numpy as np
import pandas as pd
import pywt
import config
from scipy.fft import fft, fftfreq


# ============= Time Domain Features =============

def mav(x):
    """Mean Absolute Value"""
    return np.mean(np.abs(x))

def var(x):
    """Variance"""
    return np.var(x)

def rms(x):
    """Root Mean Square"""
    return np.sqrt(np.mean(x**2))

def ssc(x, threshold=0):
    """Slope Sign Change"""
    diff = np.diff(x)
    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    return sign_changes

def zc(x, threshold=0):
    """Zero Crossing"""
    return np.sum(np.diff(np.sign(x)) != 0)

def wl(x):
    """Wave Length"""
    return np.sum(np.abs(np.diff(x)))

def ar5(x):
    """5th order Autoregressive coefficients"""
    # Burg method for AR coefficients
    r = np.correlate(x, x, mode='full')[len(x)-1:]
    r = r[:6] / r[0]
    
    # Levinson-Durbin recursion
    ar = np.zeros(5)
    e = r[0]
    for k in range(5):
        lambda_k = r[k+1] - np.sum(ar[:k] * r[k:0:-1])
        ar_new = np.zeros(k+2)
        ar_new[k+1] = lambda_k / e
        ar_new[:k+1] = ar[:k+1] - ar_new[k+1] * ar[k::-1]
        ar = ar_new
        e = e * (1 - ar_new[k+1]**2)
    
    return ar[1:]

def cc(x, n_coeff=5):
    """Cepstral Coefficients"""
    spectrum = np.abs(fft(x))
    log_spectrum = np.log(spectrum[:len(spectrum)//2] + 1e-10)
    cepstrum = np.real(fft(log_spectrum))
    return cepstrum[:n_coeff]


# ============= Frequency Domain Features =============

def mnf(x, fs=1000):
    """Mean Frequency"""
    freqs = fftfreq(len(x), 1/fs)
    psd = np.abs(fft(x))**2
    
    # Only positive frequencies
    idx = freqs > 0
    freqs = freqs[idx]
    psd = psd[idx]
    
    return np.sum(freqs * psd) / np.sum(psd)

def mdf(x, fs=1000):
    """Median Frequency"""
    freqs = fftfreq(len(x), 1/fs)
    psd = np.abs(fft(x))**2
    
    # Only positive frequencies
    idx = freqs > 0
    freqs = freqs[idx]
    psd = psd[idx]
    
    cumsum = np.cumsum(psd)
    median_idx = np.where(cumsum >= cumsum[-1] / 2)[0][0]
    return freqs[median_idx]


# ============= Time-Frequency Features =============

def wtwl(x, wavelet='db4', level=4):
    """Wavelet Transform Waveform Length"""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    detail = coeffs[-1]  # Last detail coefficient
    return np.sum(np.abs(np.diff(detail)))

def wtvar(x, wavelet='db4', level=4):
    """Wavelet Transform Variance"""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    detail = coeffs[-1]
    return np.var(detail)

def wtmav(x, wavelet='db4', level=4):
    """Wavelet Transform Mean Absolute Value"""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    detail = coeffs[-1]
    return np.mean(np.abs(detail))


# ============= Feature Extraction =============

def extract_window_features(window, fs=1000):
    """Extract all features from a single window"""
    features = {}
    
    # Time domain
    features['MAV'] = mav(window)
    features['VAR'] = var(window)
    features['RMS'] = rms(window)
    features['SSC'] = ssc(window)
    features['ZC'] = zc(window)
    features['WL'] = wl(window)
    
    # AR coefficients
    ar_coefs = ar5(window)
    for i, coef in enumerate(ar_coefs):
        features[f'AR{i+1}'] = coef
    
    # Cepstral coefficients
    cc_coefs = cc(window)
    for i, coef in enumerate(cc_coefs):
        features[f'CC{i+1}'] = coef
    
    # Frequency domain
    features['MNF'] = mnf(window, fs)
    features['MDF'] = mdf(window, fs)
    
    # Time-frequency
    features['WTWL'] = wtwl(window)
    features['WTVAR'] = wtvar(window)
    features['WTMAV'] = wtmav(window)
    
    return features


def create_features(input_path=None, output_path=None):

    input_path = config.DATA_DIR / "preprocessed_aux_windows.pkl"

    output_path = config.DATA_DIR / "features_aux_windows.pkl"
    
    # Load preprocessed windowed data
    df_windowed = pd.read_pickle(input_path)
    print(f"Loaded {len(df_windowed)} windows with {len(df_windowed.columns)} channels")
    
    # Extract features for each window and channel
    features_list = []
    for idx in range(len(df_windowed)):
        window_features = {}
        
        for ch in df_windowed.columns:
            signal_window = df_windowed.loc[idx, ch]
            ch_features = extract_window_features(signal_window, fs=1000)
            
            # Prefix with channel name
            for feat_name, feat_val in ch_features.items():
                window_features[f"{ch}_{feat_name}"] = feat_val
        
        features_list.append(window_features)
    
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Save
    features_df.to_pickle(output_path)
    print(f"Extracted {features_df.shape[1]} features from {len(features_df)} windows → {output_path}")
    
    return features_df 