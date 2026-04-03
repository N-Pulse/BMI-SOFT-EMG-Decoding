# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from .SNR import SNR



# ================================================================
# 1. Section: Functions
# ================================================================
def known_noise(
    data_df,
    sig_dt: tuple[int, int],
    noi_dt: tuple[int, int],
) -> SNR:
    signal = data_df.loc[(data_df['Time'] >= sig_dt[0]) & (data_df['Time'] < sig_dt[1])].copy()
    noise  = data_df.loc[(data_df['Time'] >= noi_dt[0]) & (data_df['Time'] < noi_dt[1])].copy()

    snr_list = []
    snr_dict = {}
    for col_name, data in signal.items():
        if "Channel" in col_name:
            s = signal[col_name].values
            n = noise[col_name].values

            snr = compute_snr(s, n)
            snr_list.append(snr)
            snr_dict[col_name] = snr

    snr_mean = float(np.mean(snr_list))

    return SNR(snr_dict, snr_mean)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def compute_snr(
    signal: np.ndarray,
    noise: np.ndarray
) -> float:
    rms_signal = compute_rms(signal)
    rms_noise = compute_rms(noise)

    snr = 20*np.log10(rms_signal/rms_noise)
    return snr

def compute_rms(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal**2))
    return rms
