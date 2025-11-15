from data_loading_xdf import load_emg_bids, find_bids_emg_files, get_emg_channels
from trigger_to_label import map_triggers_to_labels
import argparse
import sys
import os

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
        
        print(f"✅ Successfully loaded:")

        return labeled_data
    
    except Exception as e:
        print(f"❌ Error loading EMG data: {e}")
        return None

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Load EMG data from BIDS format')
    
    # Required argument
    parser.add_argument('--root', required=True, help='Root path to BIDS dataset')
    
    # Optional arguments with defaults
    parser.add_argument('--subject', default='P005', help='Subject ID (default: P005)')
    parser.add_argument('--session', default='S002', help='Session ID (default: S002)')
    parser.add_argument('--task', default='Default', help='Task name (default: Default)')
    parser.add_argument('--run', default='001_eeg_up', help='Run identifier (default: 001_eeg_up)')
    
    # Additional options
    parser.add_argument('--list-subjects', action='store_true', help='List available subjects')
    parser.add_argument('--output', help='Output file to save loaded data')
    
    args = parser.parse_args()
    
    # List subjects if requested
    if args.list_subjects:
        try:
            subjects = find_bids_emg_files(args.root)
            print("Available subjects:")
            for sub in subjects:
                print(f"  - {sub}")
            return
        except Exception as e:
            print(f"Error listing subjects: {e}")
            return
    
    # Load EMG data
    labeled_emg = load_emg_data(
        repo_root=args.root,
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run
    )
    
    if labeled_emg is not None:
        print(f"\n🎉 Data loaded successfully!")
        
        # Save to file if output path provided
        if args.output:
            try:
                import pickle
                with open(args.output, 'wb') as f:
                    pickle.dump({'labeled_emg': labeled_emg}, f)
                print(f"💾 Data saved to: {args.output}")
            except Exception as e:
                print(f"Warning: Could not save data: {e}")
        
        return labeled_emg
    else:
        print("💥 Failed to load data")
        sys.exit(1)


if __name__ == "__main__":
    main()