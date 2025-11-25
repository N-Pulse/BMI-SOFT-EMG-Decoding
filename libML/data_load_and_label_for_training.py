import argparse
import sys
import os
import re
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libML.data_loading_xdf import load_emg_bids, find_bids_emg_files, get_emg_channels
from libML.trigger_to_label import map_triggers_to_labels, convert_labels_to_dof_dict

def get_emg_labels_from_path(data_dir, subject="P005", session="S002", task="Default", run="001_eeg_up"):
    """
    Load EMG data and labels from a BIDS-compliant directory.
    Returns: X (samples x channels), y (labels)
    """
    streams, header = load_emg_bids(data_dir, subject=subject, session=session, task=task, run=run)

    emg_channels = get_emg_channels(streams)
    trigger_labels = streams[0]

    emg_labeled = map_triggers_to_labels(emg_channels, trigger_labels)

    return emg_labeled


def load_single_emg_file(repo_root, strength_and_speed=False, subject="P005", session="S002", task="Default", run="001_eeg_up"):
    """
    Load EMG data and labels from a BIDS-compliant directory.
    """
    try:
        print(f"Loading EMG data for subject {subject}, session {session}, task {task}, run {run}")
        
        streams, header = load_emg_bids(repo_root, subject=subject, session=session, task=task, run=run)
        
        if not streams:
            raise ValueError(f"No EMG data found for the specified parameters")
        
        emg_channels = get_emg_channels(streams)

        trigger_stream = streams[0]
        labeled_data = map_triggers_to_labels(emg_channels, trigger_stream)

        # Try with explicit dtype conversion
        try:
            X_raw = np.array(labeled_data['time_series'], dtype=np.float32)
        except OverflowError:
            print("Warning: Overflow in time_series, trying float64")
            X_raw = np.array(labeled_data['time_series'], dtype=np.float64)
        
        try:
            y_raw = np.array(labeled_data['labels'], dtype=np.int32)
        except OverflowError:
            print("Warning: Overflow in labels, trying int64")
            y_raw = np.array(labeled_data['labels'], dtype=np.int64)
            
        timestamps = np.array(labeled_data['time_stamps'], dtype=np.float64)

        print(f"Successfully loaded.")

        y_raw_dict = convert_labels_to_dof_dict(y_raw, strength_and_speed)

        data_dict = {
            'X': X_raw,
            'y': y_raw_dict,
            't': timestamps,
            'sub': subject,
            'ses': session,
            'task': task,
            'run': run
        }
    
        return data_dict
    
    except Exception as e:
        print(f"Error loading EMG data: {e}")
        import traceback
        traceback.print_exc()
        return None
    

def load_emg_data(repo_root, strength_and_speed=False, **filters):
    """
    Loads all EMG data files from a BIDS-compliant dataset that match
    the provided filters.

    Args:
        repo_root (str): Root path to BIDS dataset.
        strength_and_speed (bool): Include DoF describing strength and speed of the movement
        **filters: Keyword arguments to filter by.
            Keys must be BIDS components: 'subject', 'session', 'task', 'run'.
            Values can be a single string or a list of strings.
            If a key is omitted, all values for that component are loaded.

    Example:
        # Load all data
        all_data = load_emg_data(repo_root)
        
        # Load specific subjects
        data_p1_p2 = load_emg_data(repo_root, subject=['P001', 'P002'])
        
        # Load one subject, one session
        data_p1_s1 = load_emg_data(repo_root, subject='P001', session='S001')
        
        # Load two runs for all subjects
        data_runs = load_emg_data(repo_root, run=['001_eeg_up', '002_eeg_down'])
    """
    
    # 1. Normalize filters: Ensure all filter values are lists
    processed_filters = {}
    for key, value in filters.items():
        if isinstance(value, str):
            processed_filters[key] = [value] # Convert single string to list
        elif value is not None:
            processed_filters[key] = value # Already a list
            
    print(f"Starting data load with filters: {processed_filters}")

    # 2. Compile BIDS regex to parse filenames
    # This pattern captures sub, ses, task, and run from a path
    bids_pattern = re.compile(
        r'sub-([a-zA-Z0-9]+)'           # Group 1: subject
        r'(?:_ses-([a-zA-Z0-9]+))?'     # Group 2: session (optional)
        r'(?:_task-([a-zA-Z0-9]+))?'    # Group 3: task (optional)  
        r'(?:_run-([a-zA-Z0-9_\-]+))?'  # Group 4: run (optional) - now allows underscores AND hyphens
    )

    # 3. Walk the directory and find all files that match the BIDS pattern
    discovered_files = []
    for root, dirs, files in os.walk(repo_root):
        for file in files:
            # Look for common data file extensions
            if file.endswith(('.vhdr', '.eeg', '.vmrk', '.edf', '.bdf', '.xdf', '.gdf', '.set')):
                print(os.path.join(root, file))
                match = bids_pattern.search(file) #os.path.join(root, file)
                if match:
                    discovered_files.append(match)

    if not discovered_files:
        print("Warning: No BIDS-compatible files found in repo_root.")
        return []

    # 4. Filter the discovered files
    all_data_dicts = []
    loaded_file_keys = set() # Avoid loading .vhdr, .eeg, .vmrk as 3 separate files

    for match in discovered_files:
        # Create a unique key for this file's components
        file_key = match.group(0) # e.g., "sub-P005_ses-S002_task-Default_run-001_eeg_up"
        if file_key in loaded_file_keys:
            continue
        
        components = {
            'subject': match.group(1),
            'session': match.group(2), # Will be None if not found
            'task': match.group(3),    # Will be None if not found
            'run': match.group(4)      # Will be None if not found
        }
        
        # Check this file against the user's filters
        keep_file = True
        for key, allowed_values in processed_filters.items():
            file_value = components.get(key)
            if file_value not in allowed_values:
                keep_file = False
                break # Mismatch, skip this file

        if keep_file:
            # This file matches! Add its key to avoid duplicates.
            loaded_file_keys.add(file_key)
            
            # Prepare arguments for load_single_emg_file
            # We only pass components that were *actually found* in the filename.
            # If a component is None, it's not passed, so the
            # load_single_emg_file function will use its *own* default.
            load_args = {'repo_root': repo_root, 'strength_and_speed': strength_and_speed}
            for key, val in components.items():
                if val is not None:
                    load_args[key] = val
                    
            # 5. Load the matching file
            data_dict = load_single_emg_file(**load_args)
            
            if data_dict:
                all_data_dicts.append(data_dict)

    print(f"\nLoad complete. Successfully loaded {len(all_data_dicts)} files.")
    return all_data_dicts