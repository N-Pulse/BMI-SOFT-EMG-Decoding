import numpy as np

def compute_rms(signal):
    '''
    signal: array of signal
    '''
    rms = np.sqrt(np.mean(signal**2))
    # print(f"RMS: {rms}")
    return rms

def compute_snr(signal, noise):
    '''
    signal: array of signal
    noise: array of baseline noise
    '''
    rms_signal = compute_rms(signal)
    rms_noise = compute_rms(noise)

    # print(f"RMS signal: {rms_signal}, RMS noise: {rms_noise}")

    snr = 20*np.log10(rms_signal/rms_noise)
    return snr