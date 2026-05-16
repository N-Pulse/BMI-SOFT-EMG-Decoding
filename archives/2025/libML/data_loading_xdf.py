import os
import re
import mne
import pandas as pd
import numpy as np
import pyxdf

def find_bids_emg_files(data_dir, subject, session=None, task=None, run=None):
    """
    Return list of (xdf_path, events_path, json_path) for one subject/session/task.
    data_dir must include data/raw
    """
    #subj_dir = os.path.join(bids_root, "data")
    #subj_dir = os.path.join(subj_dir, "raw")
    subj_dir = os.path.join(data_dir, f"sub-{subject}")
    if session:
        subj_dir = os.path.join(subj_dir, f"ses-{session}")
    #if task:
    #    subj_dir = os.path.join(subj_dir, f"task-{task}")
    #if run:
    #    subj_dir = os.path.join(subj_dir, f"run-{run}")
    emg_dir = os.path.join(subj_dir, "emg")
    #print(emg_dir)
    if not os.path.exists(emg_dir):
        raise FileNotFoundError(f"No EMG folder for subject {subject}, session {session}")

    pattern = f"sub-{subject}"
    if session:
        pattern += f"_ses-{session}"
    if task:
        pattern += f"_task-{task}"
    if run:
        pattern += f"_run-{run}"
    pattern += ".xdf"
    
    files = [f for f in os.listdir(emg_dir) if re.match(pattern, f)]
    return [os.path.join(emg_dir, f) for f in files]

def load_emg_bids(data_dir, subject, session=None, task=None, run=None, label_column="event_label"):
    """
    Load EMG data and labels from a BIDS-compliant directory.
    Returns: X (samples x channels), y (labels)
    """
    xdf_files = find_bids_emg_files(data_dir, subject, session, task, run)
    if len(xdf_files) == 0:
        raise FileNotFoundError(f"No EMG XDF found for task={task}")

    xdf_path = xdf_files[0]  # assuming one run
    base_prefix = xdf_path.replace("_emg.xdf", "")
    events_path = base_prefix + "_events.tsv"

    # --- Load EMG signal ---
    streams, header = pyxdf.load_xdf(xdf_path)

    return streams, header

def get_emg_channels(streams):
    """
    Extract EMG channels from the list of streams.
    
    Args:
        streams: List of data streams loaded from XDF file.
    """
    trigger_labels = streams[0]
    all_channels = streams[1]

    emg_data = all_channels['time_series']
    timestamps = all_channels['time_stamps']

    try:
        channels = all_channels['info']['desc'][0]['channels'][0]['channel']
        channel_names = [ch['label'][0] for ch in channels]
    except KeyError:
        channel_names = [f'Channel_{i+1}' for i in range(emg_data.shape[1])]
    
    # Extract channel names from the info structure
    try:
        channels = all_channels['info']['desc'][0]['channels'][0]['channel']
        channel_names = [ch['label'][0] for ch in channels]
    except KeyError as e:
        print(f"Error extracting channel names: {e}")
        channel_names = [f'Channel_{i+1}' for i in range(emg_data.shape[1])]
    
    # Determine data orientation
    if emg_data.shape[1] == len(channel_names):
        data_for_plotting = emg_data
    else:
        data_for_plotting = emg_data.T
    
    # Get the last 6 channels EXCLUDING the very last one (trigger channel)
    last_6_indices = range(len(channel_names)-7, len(channel_names)-1)
    last_6_names = [channel_names[i] for i in last_6_indices]
    
    # Extract the time series data for these 6 channels
    last_6_time_series = data_for_plotting[:, last_6_indices]
    last_6_timestamps = timestamps[:]
    
    # Return all the data
    emg_channels = {
        'channel_names': last_6_names,
        'time_series': last_6_time_series,
        'time_stamps': last_6_timestamps,
        'excluded_trigger': channel_names[-1]
    }

    return emg_channels

def load_csv_file(data_path):
    data_df = pd.read_csv(data_path, delimiter=',', skiprows=4)
    # cols = data_df.iloc[0]
    # print(cols)
    # data_df.drop(index=0, inplace=True)
    # data_df.columns = cols
    print(data_df)
    return data_df

def load_xdf_file(data_path):
    streams, header = pyxdf.load_xdf(data_path)
    emg_channels = get_emg_channels(streams)
    return emg_channels
    