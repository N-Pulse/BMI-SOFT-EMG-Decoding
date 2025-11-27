"""
Module for nPulse EMG signal processing.

This module maps the trigger signals to corresponding action labels that can be used
as training targets for machine learning models.

The decoding of the trigger labels relies on the trigger system protocol as of the 19.11.2025.
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
        1.  Thumb flexed/rest/extended  -->  0/1/2
        2.  Index flexed/rest/extended  -->  0/1/2
        3.  Middle flexed/rest/extended -->  0/1/2
        4.  Ring flexed/rest/extended   -->  0/1/2
        5.  Little flexed/rest/extended -->  0/1/2
        6.  Supination  Palm facing: Up/Side/Down  -->  0/1/2
        7.  Wrist angle Palm facing down then: Up(-90)/Straight(0)/Down(90)  -->  0/1/2
        8.  Thumb Abduction --> extended dorsal / rest / extended palmar --> 0/1/2
        9.  Strength --> not specified / low / medium / high --> 0/1/2/3
        10. Speed --> not specified / low / medium / high --> 0/1/2/3

    Label is encoded in a 10-digit value (int)
    Label = -1 is for non interesting data

    """
    # Convert to numpy array with safe dtype first
    if not isinstance(trigger_labels, np.ndarray):
        trigger_labels = np.array(trigger_labels, dtype=np.int64)
    else:
        trigger_labels = trigger_labels.astype(np.int64)
    
    new_labels = np.full_like(trigger_labels, -1, dtype=np.int64)
    
    for i in range(len(trigger_labels)):
        label = trigger_labels[i]

        # Special codes
        if label == 9701 or label == 9702:
            if i != 0:
                new_labels[i] = -1  # or new_labels[i-1] if you want to carry forward
            else:  # in case we start the recording in resting state
                new_labels[i] = -1
            continue
        if label in [8888, 9999, 8899]:
            new_labels[i] = -1
            continue

        # Get codes
        phase_label = label // 10000
        arm_label = (label - phase_label * 10000) // 1000  # don't care actually
        baseline_label = (label - phase_label * 10000 - arm_label * 1000) // 100
        movement_label = label - phase_label * 10000 - arm_label * 1000 - baseline_label * 100
        
        new_label = 0

        # No movements
        if phase_label not in [3, 4]:  # move and return
            new_labels[i] = -1  # not interesting for training
            continue

        # Disregarded movements
        if movement_label in []:
            new_labels[i] = -1
            continue

        # Movements
        if phase_label in [3, 4]:
            # Use proper base-10 encoding instead of floating point exponents
            # Each digit gets its own place value (10^0, 10^1, 10^2, etc.)
            
            # Thumb flex/rest/ext --> 0/1/2 (1st digit)
            if movement_label in [3, 4, 5, 6, 10, 15, 16, 22]:  # flexed
                digit = 0
            elif movement_label in [7, 8, 11, 12, 13, 14, 17, 18, 19, 20, 21, 23, 24, 25, 26]:  # rest
                digit = 1
            else:  # movement_label in [1, 2, 9, 27] - extended
                digit = 2
            new_label += digit * (10 ** 0)

            # Index flex/rest/ext --> 0/1/2 (2nd digit)
            if movement_label in [3, 4, 5, 6, 8, 15, 16, 21]:  # flexed
                digit = 0
            elif movement_label in [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 22, 23, 24, 25, 27]:  # rest
                digit = 1
            else:  # movement_label in [1, 2, 7, 26] - extended
                digit = 2
            new_label += digit * (10 ** 1)

            # Middle flex/rest/ext --> 0/1/2 (3rd digit)
            if movement_label in [3, 4, 5, 6, 8, 15, 16, 20]:  # flexed
                digit = 0
            elif movement_label in [9, 10, 11, 12, 13, 14, 17, 18, 19, 21, 22, 23, 24, 26, 27]:  # rest
                digit = 1
            else:  # movement_label in [1, 2, 7, 25] - extended
                digit = 2
            new_label += digit * (10 ** 2)

            # Ring flex/rest/ext --> 0/1/2 (4th digit)
            if movement_label in [3, 4, 5, 6, 8, 15, 16, 19]:  # flexed
                digit = 0
            elif movement_label in [9, 10, 11, 12, 13, 14, 17, 18, 20, 21, 22, 23, 25, 26, 27]:  # rest
                digit = 1
            else:  # movement_label in [1, 2, 7, 24] - extended
                digit = 2
            new_label += digit * (10 ** 3)

            # Pinky flex/rest/ext --> 0/1/2 (5th digit)
            if movement_label in [3, 4, 5, 6, 8, 15, 16, 18]:  # flexed
                digit = 0
            elif movement_label in [9, 10, 11, 12, 13, 14, 17, 19, 20, 21, 22, 24, 25, 26, 27]:  # rest
                digit = 1
            else:  # movement_label in [1, 2, 7, 23] - extended
                digit = 2
            new_label += digit * (10 ** 4)

            # Supination codes --> 0/1/2 (6th digit)
            if baseline_label == 1:  # palm/fist up
                digit = 0
            elif baseline_label == 2:  # palm/fist side
                digit = 1
            else:  # baseline_label == 3 - palm/fist down
                digit = 2
            new_label += digit * (10 ** 5)

            # Wrist codes (7th digit)
            if movement_label in [13, 14]:
                digit = 0  # Up(-90)
            elif movement_label in [11, 12]:
                digit = 2  # Down(90)
            else:  # Straight(0)
                digit = 1
            new_label += digit * (10 ** 6)

            # Thumb abduction (8th digit)
            if movement_label in [1, 2, 9, 27]:  # dorsal
                digit = 0
            elif movement_label in [7, 8, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26]:  # rest
                digit = 1
            else:  # movement_label in [3, 4, 5, 6, 10, 15, 16, 17, 22] - palmar
                digit = 2
            new_label += digit * (10 ** 7)

            # Strength (9th digit)
            if movement_label in [5, 12, 13]:  # normal
                digit = 2
            elif movement_label in [6, 11, 14]:  # high
                digit = 3
            elif movement_label in []:  # low
                digit = 1
            else:  # not specified
                digit = 0
            new_label += digit * (10 ** 8)

            # Speed (10th digit)
            if movement_label in [1, 3]:  # low
                digit = 1
            elif movement_label in [2, 4]:  # high
                digit = 3
            elif movement_label in []:  # normal
                digit = 2
            else:  # not specified
                digit = 0
            new_label += digit * (10 ** 9)

        new_labels[i] = new_label
    
    return new_labels

def convert_labels_to_dof_dict(y, strength_and_speed: bool):
    """
    Convert the final label array y to a dictionary of 8 degrees of freedom.
    
    Assumes y contains the 8-digit encoded labels from your original mapping.
    
    Returns: dict with keys 'dof_1' to 'dof_8' representing each degree of freedom
    """
    # Initialize arrays for each degree of freedom
    dof_1 = np.full(len(y), -1, dtype=int)  # Thumb flexion
    dof_2 = np.full(len(y), -1, dtype=int)  # Index flexion  
    dof_3 = np.full(len(y), -1, dtype=int)  # Middle flexion
    dof_4 = np.full(len(y), -1, dtype=int)  # Ring flexion
    dof_5 = np.full(len(y), -1, dtype=int)  # Little flexion
    dof_6 = np.full(len(y), -1, dtype=int)  # Supination
    dof_7 = np.full(len(y), -1, dtype=int)  # Wrist flexion
    dof_8 = np.full(len(y), -1, dtype=int)  # Thumb abduction
    dof_9 = np.full(len(y), -1, dtype=int)  # Strength
    dof_10 = np.full(len(y), -1, dtype=int)  # Speed
    
    for i, label in enumerate(y):
        if label == -1:
            # Skip invalid labels
            continue
            
        # Decode the 8-digit number back to individual DoFs
        # Assuming the encoding is: dof_1 * 10^0 + dof_2 * 10^1 + ... + dof_8 * 10^7
        temp_label = label
        dof_1[i] = temp_label % 10
        temp_label //= 10
        dof_2[i] = temp_label % 10
        temp_label //= 10
        dof_3[i] = temp_label % 10
        temp_label //= 10
        dof_4[i] = temp_label % 10
        temp_label //= 10
        dof_5[i] = temp_label % 10
        temp_label //= 10
        dof_6[i] = temp_label % 10
        temp_label //= 10
        dof_7[i] = temp_label % 10
        temp_label //= 10
        dof_8[i] = temp_label % 10
        if strength_and_speed:
            temp_label //= 10
            dof_9[i] = temp_label % 10
            temp_label //= 10
            dof_10[i] = temp_label % 10
    if strength_and_speed:
        return {
            'dof_1': dof_1,  # Thumb flexion (0: flexed, 1: rest, 2: extended)
            'dof_2': dof_2,  # Index flexion (0: flexed, 1: rest, 2: extended)
            'dof_3': dof_3,  # Middle flexion (0: flexed, 1: rest, 2: extended)
            'dof_4': dof_4,  # Ring flexion (0: flexed, 1: rest, 2: extended)
            'dof_5': dof_5,  # Little flexion (0: flexed, 1: rest, 2: extended)
            'dof_6': dof_6,  # Supination (0: up, 1: side, 2: down)
            'dof_7': dof_7,  # Wrist angle (0: up, 1: straight, 2: down)
            'dof_8': dof_8,  # Thumb abduction (0: dorsal, 1: rest, 2: palmar)
            'dof_9': dof_9,  # Strength (0: NaN, 0: Low, 1: Medium, 2: High)
            'dof_10': dof_10,  # Speed (0: NaN, 0: Low, 1: Medium, 2: High)
            'original_labels': y  # Keep the original encoded labels
        }
    else:
        return {
            'dof_1': dof_1,  # Thumb flexion (0: flexed, 1: rest, 2: extended)
            'dof_2': dof_2,  # Index flexion (0: flexed, 1: rest, 2: extended)
            'dof_3': dof_3,  # Middle flexion (0: flexed, 1: rest, 2: extended)
            'dof_4': dof_4,  # Ring flexion (0: flexed, 1: rest, 2: extended)
            'dof_5': dof_5,  # Little flexion (0: flexed, 1: rest, 2: extended)
            'dof_6': dof_6,  # Supination (0: up, 1: side, 2: down)
            'dof_7': dof_7,  # Wrist angle (0: up, 1: straight, 2: down)
            'dof_8': dof_8,  # Thumb abduction (0: dorsal, 1: rest, 2: palmar)
            'original_labels': y  # Keep the original encoded labels
        }