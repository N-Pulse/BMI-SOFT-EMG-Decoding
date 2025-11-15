"""
Module for nPulse EMG signal processing.

This module maps the trigger signals to corresponding action labels that can be used
as training targets for machine learning models.
"""

import numpy as np

def map_triggers_to_labels(emg_channels, trigger_labels):
    """
    Map trigger signals to action labels.
    It also deletes the data that was sampled before the first trigger.

    Args:
        emg_channels: EMG data channels (will be reformated)
        trigger_channel: Trigger channel (0 for each timestamp) but gives us length of data 
                         and corresponding timestamps
        trigger_labels: Trigger labels following the Data Acquisition Protocol

    Returns:
        train_labels: List of labels (format needed for training) for each timestamp after the first trigger
        emg_clean: Cleand EMG channels starting from the first trigger
    """
    emg_channels = emg_channels.copy()
    trigger_labels = trigger_labels.copy()

    # First step: Find the first trigger index and trim data preceding it
    emg_clean = clean_emg_from_index(emg_channels, trigger_labels)

    # Second step: Define Map that transforms triggers to labels
    # Create a copy of the time_series array to avoid modifying the original
    mapped_trigger_series = map_protocol_to_label(trigger_labels["time_series"].copy())
    
    # Create a new dictionary with mapped labels but keep the original structure
    mapped_trigger_labels = {
        "time_stamps": trigger_labels["time_stamps"],
        "time_series": mapped_trigger_series
    }

    # Third step: Create train_labels list --> label each timestamp
    emg_labeled = extend_labels(emg_clean, mapped_trigger_labels)

    return emg_labeled

def clean_emg_from_index(emg_channels, trigger_labels):
    """
    Find the index of the first trigger in the trigger channel.

    Args:
        emg_channels: EMG data channels
        trigger_labels: Trigger labels following the Data Acquisition Protocol

    Returns:
        emg_channels, trigger_channel: Filtered EMG and trigger channels starting from first trigger
    """
    ts = np.array(emg_channels["time_stamps"])
    if ts.size == 0:
        print('No timestamps in channels to filter.')
    else:
        first_label_ts = trigger_labels["time_stamps"][0]
        mask = ts >= first_label_ts
        removed = np.count_nonzero(~mask)
        # Update timestamps
        emg_channels["time_stamps"] = ts[mask].tolist()
        # Update corresponding time_series. Handle common orientations:
        emg_data = np.array(emg_channels["time_series"])
        # If rows correspond to timestamps (n_samples, n_channels)
        if emg_data.shape[0] == ts.size:
            emg_channels["time_series"] = emg_data[mask].tolist()
        # If columns correspond to timestamps (n_channels, n_samples)
        elif emg_data.ndim == 2 and emg_data.shape[1] == ts.size:
            emg_channels["time_series"] = emg_data[:, mask].tolist()
        else:
            # Fallback: try to filter rows; if that fails, leave emg_data as-is
            try:
                emg_channels["time_series"] = emg_data[mask].tolist()
            except Exception as e:
                print(f'Could not apply mask to time_series automatically: {e}')

    return emg_channels

def extend_labels(emg_channels, trigger_labels):
    """
    Extend the data acquisition protocol labels to every timestamp

    Args:
        emg_channels: EMG data channels
        trigger_labels: Trigger labels following the Data Acquisition Protocol

    Returns:
        emg_channels, trigger_channel: Filtered EMG and trigger channels starting from first trigger
    """
    # Expect trigger_labels to be streams[0] and emg_channels to be produced earlier
    trigger_ts = trigger_labels["time_stamps"]
    trigger_vals = trigger_labels["time_series"]
    # Flatten and decode bytes if necessary
    if trigger_vals.ndim > 1:
        trigger_vals = trigger_vals.flatten()
    trigger_vals = [v.decode() if isinstance(v, (bytes, bytearray)) else v for v in trigger_vals]

    emg_ts = np.array(emg_channels.get("time_stamps", []))

    # Prepare labels array aligned to emg_ts
    if trigger_ts.size == 0 or emg_ts.size == 0:
        # Nothing to map; create same-length None list
        emg_channels["labels"] = [None] * emg_ts.size
        print("No trigger timestamps or no EMG timestamps — created empty labels list")
    else:
        # Find for each emg timestamp the index of the latest trigger <= ts
        inds = np.searchsorted(trigger_ts, emg_ts, side="right") - 1
        labels = []
        for idx in inds:
            if idx < 0:
                # Timestamp occurs before the first trigger
                labels.append(None)
            else:
                labels.append(trigger_vals[idx])
        emg_channels["labels"] = labels

    return emg_channels

def map_protocol_to_label(trigger_labels):
    """
    Dummy version : Not taking into account initial hand state, only final state of the action

    For each protocol movement, label 8 degrees of freedom :
        1. Thumb flexed/rest/extended  -->  0/1/2
        2. Index flexed/rest/extended  -->  0/1/2
        3. Middle flexed/rest/extended -->  0/1/2
        4. Ring flexed/rest/extended   -->  0/1/2
        5. Little flexed/rest/extended -->  0/1/2
        6. Supination  Palm facing: Up/Side/Down  -->  0/1/2
        7. Wrist angle Palm facing down then: Up(-90)/Straight(0)/Down(90)  -->  0/1/2
        8. Thumb Abduction --> extended dorsal / rest / extended palmar --> 0/1/2

    Label is encoded in a 8-digit value, its order in the list corresponding to its digit + 1 (1. --> 10e0, 2 --> 10e1, ...)
    Label = -1 is for non interesting data

    """
    
    for i in range(len(trigger_labels)):
        label = trigger_labels[i]

        # Special codes
        if label == 9701 or label == 9702 :
            if i != 0 :
                trigger_labels[i] = -1 #trigger_labels[i-1] #
            if i == 0 : # in case we start the recording in resting state, which should not happen
                trigger_labels[i] = -1 #00000000 # to be defined 
            continue
        if label in [8888, 9999, 8899] :
            trigger_labels[i] = -1
            continue

        # Get codes
        phase_label = label//10000
        arm_label = (label-phase_label*10000)//1000 # dont care actually
        baseline_label = (label-phase_label*10000-arm_label*1000)//100
        movement_label = label-phase_label*10000-arm_label*1000-baseline_label*100
        #print(f"Label: {label}, Phase: {phase_label}, Arm: {arm_label}, Baseline: {baseline_label}, Movement: {movement_label}")
        new_label = 0

        # No movements
        if phase_label not in [3,4] : # move and return
            trigger_labels[i] = -1 # not interesting for training
            continue

        # Disregarded movements
        if movement_label in []:
            trigger_labels[i] = -1
            continue

        # Movements
        if phase_label in [3,4] :
            # Thumb flex/rest/ext --> 0/1/2
            if movement_label in [3,4,5,6,10,15,16,22]: # flexed
                new_label += 0 * 10e-1
            if movement_label in [7,8,11,12,13,14,17,18,19,20,21,23,24,25,26]: # rest
                new_label += 1 * 10e-1
            if movement_label in [1,2,9,27]: # extended
                new_label += 2 * 10e-1
            
            # Index flex/rest/ext --> 0/1/2
            if movement_label in [3,4,5,6,8,15,16,21]: # flexed
                new_label += 0 * 10e0
            if movement_label in [9,10,11,12,13,14,17,18,19,20,22,23,24,25,27]: # rest
                new_label += 1 * 10e0
            if movement_label in [1,2,7,26]: # extended
                new_label += 2 * 10e0

            # Middle flex/rest/ext --> 0/1/2
            if movement_label in [3,4,5,6,8,15,16,20]: # flexed
                new_label += 0 * 10e1
            if movement_label in [9,10,11,12,13,14,17,18,19,21,22,23,24,26,27]: # rest
                new_label += 1 * 10e1
            if movement_label in [1,2,7,25]: # extended
                new_label += 2 * 10e1
            
            # Ring flex/rest/ext --> 0/1/2
            if movement_label in [3,4,5,6,8,15,16,19]: # flexed
                new_label += 0 * 10e2
            if movement_label in [9,10,11,12,13,14,17,18,20,21,22,23,25,26,27]: # rest
                new_label += 1 * 10e2
            if movement_label in [1,2,7,24]: # extended
                new_label += 2 * 10e2
            
            # Pinky flex/rest/ext --> 0/1/2
            if movement_label in [3,4,5,6,8,15,16,18]: # flexed
                new_label += 0 * 10e3
            if movement_label in [9,10,11,12,13,14,17,19,20,21,22,24,25,26,27]: # rest
                new_label += 1 * 10e3
            if movement_label in [1,2,7,23]: # extended
                new_label += 2 * 10e3

            # Supination codes --> 0/1/2 and 6th degree of freedom
            if baseline_label == 1 : # palm/fist up
                new_label += 0 * 10e4
            if baseline_label == 2 : # palm/fist side
                new_label += 1 * 10e4
            if baseline_label == 3 : # palm/fist down
                new_label += 2 * 10e4

            # Wrist codes, Palm facing down then: Up(-90)/Straight(0)/Down(90)  -->  0/1/2
            if movement_label in [13, 14]:
                new_label += 0 * 10e5
            if movement_label in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:
                new_label += 1 * 10e5
            if movement_label in [11, 12]:
                new_label += 2 * 10e5

            # Thumb abduction --> extended dorsal / rest / extended palmar --> 0/1/2
            if movement_label in [1, 2, 9, 27]: # dorsal
                new_label += 0 * 10e6
            if movement_label in [7, 8, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26]: # rest
                new_label += 1 * 10e6
            if movement_label in [3, 4, 5, 6, 10, 15, 16, 17, 22]: # palmar 
                new_label += 2 * 10e6
        trigger_labels[i] = new_label
    return trigger_labels