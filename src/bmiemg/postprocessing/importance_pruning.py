# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier



# ================================================================
# 1. Section: Functions
# ================================================================
def prune_features(
    epochs: mne.Epochs | mne.EpochsArray,
    features: np.ndarray,
    features_funcs: list,
    clf: DecisionTreeClassifier,
    nr_to_keep: int = 7
) -> np.ndarray:
    feature_info = get_feature_df(epochs, features_funcs)
    feature_info["importance"] = clf.feature_importances_

    feature_type_importance = (
        feature_info    #type: ignore
        .groupby("feature_type", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    top_feature_types = (
        feature_type_importance
        .head(nr_to_keep)["feature_type"]
        .tolist()
    )

    selected_indices = feature_info.index[
        feature_info["feature_type"].isin(top_feature_types)
    ].to_numpy()

    return features[:, selected_indices]



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_feature_df(
    epochs: mne.Epochs | mne.EpochsArray,
    features_funcs: list,
) -> pd.DataFrame:
    rows = []

    for feature_func in features_funcs:
        for ch_name in epochs.ch_names:
            rows.append({
                "feature_type": feature_func.__name__,
                "channel": ch_name,
                "feature": f"{feature_func.__name__}_{ch_name}",
            })

    return pd.DataFrame(rows)
