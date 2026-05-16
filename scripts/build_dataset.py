"""
This scripts takes all the data we have and builds the naive Dataset
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne
import contextlib
import io
import gc

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from typing import cast
from pathlib import Path
from datetime import date, datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from bmiemg.data.convert import session_load, ChannelSplitter
from bmiemg.data.epoch import (
    SignalPartitioner,
    V1_TRIGGER_MAP,
    V2_TRIGGER_MAP,
    TriggerMap,
    average_movement_duration
)
from bmiemg.preprocessing import (
    get_envelop,
    TIME_FEATURE_FUNCTIONS,
    FREQ_FEATURE_FUNCTIONS
)



# ================================================================
# 1. Section: INPUTS
# ================================================================
# Paths
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data" / "bids"

# Version relative inputs
SESSIONS_TO_IGNORE_V1: set = set([
    "sub-05_ses-04_task-Down_run-01_raw.xdf"
])
SESSIONS_TO_IGNORE_V2: set = set([])
CUTOFF_DATE = date(2025, 11, 21)

WINDOW_SIZE: float = 0.1



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def get_date_from_parents(path: Path) -> date | None:
    for parent in path.parents:
        try:
            return datetime.strptime(parent.name, "%Y-%m-%d").date()
        except ValueError:
            continue

    return None

def get_xdf_paths(ignore_set: set, period: str) -> list:
    if period.lower() == "before":
        return [
            file
            for file in DATA.rglob("*.xdf")
            if file.is_file()
            and file.name not in ignore_set
            and (folder_date := get_date_from_parents(file)) is not None
            and folder_date < CUTOFF_DATE
        ]
    else:
        return [
            file
            for file in DATA.rglob("*.xdf")
            if file.is_file()
            and file.name not in ignore_set
            and (folder_date := get_date_from_parents(file)) is not None
            and folder_date >= CUTOFF_DATE
        ]

def load_emg_sessions(
    splitter: ChannelSplitter,
    list_of_files: list[Path]
) -> list[mne.io.RawArray]:
    emg_signals = []
    for file in list_of_files:
        # 1. Load the session
        print(f"Loading {file}")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            session = session_load(file)
            bio_signal = splitter.split(session)
            signal = bio_signal.attach_annotations()
            emg_signals.append(signal["EMG"])

        del(bio_signal)
        del(signal)
        del(session)

    return emg_signals

def extract_epochs(emg_signals: list, partinioner: SignalPartitioner):
    epochs_group = []
    for emg_signal in emg_signals:
        epochs_group.append(partinioner.partition(emg_signal))

    merged_epochs = mne.concatenate_epochs(epochs_group)

    return merged_epochs

def extract_emg_features(
    X: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    feature_values = []

    for feature_func in TIME_FEATURE_FUNCTIONS:
        val = feature_func(X)
        if val.ndim != 2:
            raise ValueError(
                f"{feature_func.__name__} returned shape {val.shape}, "
                "expected (n_epochs, n_channels)."
            )
        feature_values.append(val)

    for feature_func in FREQ_FEATURE_FUNCTIONS:
        val = feature_func(X, sfreq)
        if val.ndim != 2:
            raise ValueError(
                f"{feature_func.__name__} returned shape {val.shape}, "
                "expected (n_epochs, n_channels)."
            )
        feature_values.append(val)

    features = np.concatenate(feature_values, axis=1)

    return features

def train_dt_all(
    epochs: mne.EpochsArray | mne.Epochs,
    model_name: str,
    features: np.ndarray
) -> tuple:
    model_code = epochs.event_id[model_name]
    y_binary = (y == model_code).astype(int) # 1 vs all

    clf = DecisionTreeClassifier(
        random_state=42,
    )
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        clf,
        features,
        y_binary,
        cv=cv,
        scoring="accuracy",
    )

    clf.fit(features, y_binary)

    return clf, scores

def get_feature_df(epochs: mne.Epochs | mne.EpochsArray) -> pd.DataFrame:
    rows = []

    for feature_func in TIME_FEATURE_FUNCTIONS:
        for ch_name in epochs.ch_names:
            rows.append({
                "feature_type": feature_func.__name__,
                "channel": ch_name,
                "feature": f"{feature_func.__name__}_{ch_name}",
            })

    for feature_func in FREQ_FEATURE_FUNCTIONS:
        for ch_name in epochs.ch_names:
            rows.append({
                "feature_type": feature_func.__name__,
                "channel": ch_name,
                "feature": f"{feature_func.__name__}_{ch_name}",
            })

    return pd.DataFrame(rows)



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Finding the files
    v1_files = get_xdf_paths(SESSIONS_TO_IGNORE_V1, "before")
    v2_files = get_xdf_paths(SESSIONS_TO_IGNORE_V2, "after")
    total_nr_files = len(v1_files) + len(v2_files)
    print(
        f"Found {len(v1_files)} files from V1 protocol and "
        f"found {len(v2_files)} from V2 protocol (total {total_nr_files})\n"
    )

    # 2. Loads the files
    biotech_splitter = ChannelSplitter()
    emg_signals_v1 = load_emg_sessions(biotech_splitter, v1_files)
    emg_signals_v2 = load_emg_sessions(biotech_splitter, v2_files)
    total_nr_loaded_files = len(emg_signals_v1) + len(emg_signals_v2)
    print(
        f"Finished loading the {total_nr_loaded_files} from the original "
        f"{total_nr_files} paths (success = "
        f"{(total_nr_loaded_files/total_nr_files * 100)}\n"
    )

    # 3. Extracts the epochs
    partinioner_v1 = SignalPartitioner(V1_TRIGGER_MAP)
    partinioner_v2 = SignalPartitioner(V2_TRIGGER_MAP)
    epochs_v1 = extract_epochs(emg_signals_v1, partinioner_v1)
    epochs_v2 = extract_epochs(emg_signals_v2, partinioner_v2)

    # 4, Organize into target ones
    grouped_v1_epochs = partinioner_v1.group(epochs_v1)
    grouped_v2_epochs = partinioner_v2.group(epochs_v2)
    grouped_epochs = mne.concatenate_epochs([grouped_v1_epochs, grouped_v2_epochs])

    # 5. Cleans the epochs
    """grouped_epochs = grouped_epochs.copy().drop_bad(
        reject={"emg": 100e-5}, #type: ignore
        flat={"emg": 1e-9}      #type: ignore
    )"""

    # save should be here

    # 6. Get the envelops
    envelop_epochs = get_envelop(
        cast(mne.EpochsArray, grouped_epochs),
        window_s=WINDOW_SIZE,
    )

    # 7. Extract the features
    signal = envelop_epochs.get_data()
    y = envelop_epochs.events[:, -1]
    sfreq = envelop_epochs.info["sfreq"]
    features = extract_emg_features(signal, sfreq)
    print(
        f"Extracted {features.shape[1]} features from "
        f"{features.shape[0]} epochs\n"
    )

    # 8. Train the model with feature prunning
    models = list(envelop_epochs.event_id.keys())
    print(f"Models to try: {models}")
    for model in models:
        print(f"|----- {model} Model Training -----|")
        clf, scores = train_dt_all(envelop_epochs, model, features)

        print(f"Decision tree: {model} vs no {model}")
        print("Mean accuracy:", scores.mean())

        feature_info = get_feature_df(envelop_epochs)
        feature_info["importance"] = clf.feature_importances_

        feature_type_importance = (
            feature_info    #type: ignore
            .groupby("feature_type", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
        )

        top_feature_types = (
            feature_type_importance
            .head(7)["feature_type"]
            .tolist()
        )

        selected_indices = feature_info.index[
            feature_info["feature_type"].isin(top_feature_types)
        ].to_numpy()

        selected_features = features[:, selected_indices]
        print("Selected feature types:", top_feature_types)
        print("Original:", features.shape)
        print("Selected:", selected_features.shape)

        clf, scores = train_dt_all(envelop_epochs, model, selected_features)

        print(f"Decision tree (after pruning): {model} vs no {model}")
        print("Mean accuracy:", scores.mean())
        print()
