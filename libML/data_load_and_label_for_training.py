from data_loading_xdf import load_emg_bids, find_bids_emg_files, get_emg_channels
from trigger_to_label import map_triggers_to_labels, convert_labels_to_dof_dict
import argparse
import sys
import os
import numpy as np

def get_emg_labels_from_path(repo_root, subject="P005", session="S002", task="Default", run="001_eeg_up"):
    """
    Load EMG data and labels from a BIDS-compliant directory.
    Returns: X (samples x channels), y (labels)
    """
    streams, header = load_emg_bids(repo_root, subject=subject, session=session, task=task, run=run)

    emg_channels = get_emg_channels(streams)
    trigger_labels = streams[0]

    emg_labeled = map_triggers_to_labels(emg_channels, trigger_labels)

    return emg_labeled


def load_emg_data(repo_root, subject="P005", session="S002", task="Default", run="001_eeg_up"):
    """
    Load EMG data and labels from a BIDS-compliant directory.
    
    Args:
        repo_root (str): Root path to BIDS dataset
        subject (str): Subject ID (e.g., "P005")
        session (str): Session ID (e.g., "S002") 
        task (str): Task name (e.g., "Default")
        run (str): Run identifier (e.g., "001_eeg_up")
    
    Returns:
        labeled_data (dict): Dictionary with keys 'time_series' (EMG data) and 'labels' (corresponding labels)
    """
    try:
        print(f"Loading EMG data for subject {subject}, session {session}, task {task}, run {run}")
        
        streams, header = load_emg_bids(repo_root, subject=subject, session=session, task=task, run=run)
        
        if not streams:
            raise ValueError(f"No EMG data found for the specified parameters")
        
        emg_channels = get_emg_channels(streams)
        trigger_stream = streams[0]
        
        labeled_data = map_triggers_to_labels(emg_channels, trigger_stream)

        X_raw = np.array(labeled_data['time_series'])  # Convert to numpy array
        y_raw = np.array(labeled_data['labels'])       # Convert to numpy array
        
        timestamps = np.array(labeled_data['time_stamps'])

        print(f"Successfully loaded.")

        y_raw_dict = convert_labels_to_dof_dict(y_raw)
    
        return X_raw, y_raw_dict, timestamps
    
    except Exception as e:
        print(f"Error loading EMG data: {e}")
        return None