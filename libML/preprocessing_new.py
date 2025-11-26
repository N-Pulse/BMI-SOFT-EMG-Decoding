import numpy as np
import pandas as pd
import pyxdf
from scipy import signal


def segment_aux_windows(data, timestamps, channel_labels, window_ms=200, step_ms=100, sampling_rate=1000):
    aux_channels = ['AUX7', 'AUX8', 'AUX9', 'AUX10', 'AUX11', 'AUX12']
    aux_indices = [channel_labels.index(ch) for ch in aux_channels]
    aux_data = data[:, aux_indices]
    
    window_samples = int((window_ms / 1000.0) * sampling_rate)
    step_samples = int((step_ms / 1000.0) * sampling_rate)
    n_windows = (aux_data.shape[0] - window_samples) // step_samples + 1
    
    windowed_data = {ch: [] for ch in aux_channels}
    
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        if end > aux_data.shape[0]:
            break
        for ch_idx, ch_name in enumerate(aux_channels):
            windowed_data[ch_name].append(aux_data[start:end, ch_idx])
    
    return pd.DataFrame(windowed_data)

def segment_aux_windows_new(data, labels, window_ms=200, step_ms=100, sampling_rate=1000):
    """
    Create windowed DataFrame directly from EMG data array.
    
    Args:
        data: numpy array of shape (n_samples, n_channels) - EMG data
        window_ms: window length in milliseconds
        step_ms: step size between windows in milliseconds  
        sampling_rate: sampling rate in Hz
    
    Returns:
        pd.DataFrame: Windowed data with columns as channel indices and window indices as rows
    """
    window_samples = int((window_ms / 1000.0) * sampling_rate)
    step_samples = int((step_ms / 1000.0) * sampling_rate)
    n_windows = (data.shape[0] - window_samples) // step_samples + 1
    
    windowed_data = {ch: [] for ch in range(data.shape[1])}  # Use data.shape[1] for safety
    window_labels = {dof: [] for dof in labels.keys()}  # This creates lists for each DOF
    window_indices = []
    
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        if end > data.shape[0]:
            break

        window_label_segment = labels["original_labels"][start:end]
        unique_labels = np.unique(window_label_segment)

        if len(unique_labels) == 1 and unique_labels[0] != -1:
            # Append data for each channel
            for ch_idx in range(data.shape[1]):
                windowed_data[ch_idx].append(data[start:end, ch_idx])
                
            for dof in labels.keys():
                if dof != "original_labels":  # Skip original_labels if it's not a DOF
                    unique_dof_labels = np.unique(labels[dof][start:end])
                    if len(unique_dof_labels) == 1:
                        window_labels[dof].append(unique_dof_labels[0])
                    else:
                        # Handle mixed labels in window - use majority vote or skip
                        window_labels[dof].append(unique_dof_labels[0])  # or use mode

            window_indices.append(i)

    # Create DataFrame
    df = pd.DataFrame(windowed_data)
    
    # Add labels as columns
    for dof in labels.keys():
        if dof != "original_labels" and len(window_labels[dof]) > 0:
            df[dof + '_label'] = window_labels[dof]
    
    df['window_index'] = window_indices

    print(f"Created {len(df)} windows with unique labels.")
    print(f"Label distribution: {df['thumb_label'].value_counts() if 'thumb_label' in df.columns else 'Check column names'}")
    
    return df

def notch_filter(df, fs=1000, freq=50.0, q=30.0):
    """Remove power line interference at 50 Hz (or 60 Hz for US)"""
    b, a = signal.iirnotch(freq, q, fs)
    
    filtered_df = df.copy()
    for col in df.columns:
        # Skip non-time-series columns
        if col == 'window_index' or col == 'label':
            continue
        # Apply filter only to columns that contain arrays
        filtered_df[col] = df[col].apply(lambda x: signal.filtfilt(b, a, x) if isinstance(x, (list, np.ndarray)) and len(x) > 1 else x)
    
    return filtered_df


def passband_filter(df, fs=1000, lowcut=20.0, highcut=300.0, order=4):
    """Bandpass filter for EMG signals (20-450 Hz)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    
    filtered_df = df.copy()
    for col in df.columns:
        # Skip non-time-series columns
        if col == 'window_index' or col == 'label':
            continue
        # Apply filter only to columns that contain arrays
        filtered_df[col] = df[col].apply(lambda x: signal.filtfilt(b, a, x) if isinstance(x, (list, np.ndarray)) and len(x) > 1 else x)
    
    return filtered_df



def preprocess(file_path=None, output_path=None):

    if file_path is None:
        file_path = "./data/raw/sub-P005_ses-S002_task-Default_run-001_eeg_up.xdf"
    
    if output_path is None:
        output_path = "./data/raw_aux_windows.pkl"
    output_path_2 = "./data/preprocessed_aux_windows.pkl"
    
    # Load XDF
    streams, _ = pyxdf.load_xdf(str(file_path))
    stream = streams[1]
    
    # Extract data
    data = stream['time_series']
    timestamps = stream['time_stamps']
    sampling_rate = float(stream['info']['nominal_srate'][0])
    
    # Get channel labels
    info = stream['info']
    desc = info['desc'][0]
    
    if 'channels' in desc and desc['channels']:
        ch_list = desc['channels'][0]['channel']
        channel_labels = [ch['label'][0] for ch in ch_list]
    
    print(f"Found {len(channel_labels)} channels")
    print(f"Sample channels: {channel_labels[:10]}")
    
    # Segment windows
    windowed_df = segment_aux_windows(data, timestamps, channel_labels, sampling_rate=sampling_rate)
    
    # Save raw windows
    windowed_df.to_pickle(output_path)

    # Apply filters in order: notch first, then bandpass
    preproc_df = notch_filter(windowed_df, fs=sampling_rate)
    preproc_df = passband_filter(preproc_df, fs=sampling_rate)
    
    # Save preprocessed
    preproc_df.to_pickle(output_path_2)
    print(f"Preprocessed windows → {output_path_2}")
    
    return preproc_df 