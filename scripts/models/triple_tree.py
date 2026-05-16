"""
This generates the model that is a triple decision tree, each one being
1 vs all.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import mne
import joblib

import numpy as np

from typing import cast
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

from bmiemg.postprocessing import prune_features
from bmiemg.preprocessing.features import get_emg_features
from bmiemg.models.model_factories import DecisionTreeFactory
from bmiemg.models.target_builders import OneVsAllTargetBuilder
from bmiemg.preprocessing import (
    get_envelop,
    TIME_FEATURE_FUNCTIONS,
    FREQ_FEATURE_FUNCTIONS
)
from bmiemg.models import (
    TrainingConfig,
    ModelRegistry,
    Trainer,
    Evaluator
)



# ================================================================
# 1. Section: IMPORTS
# ================================================================
# 1.1. Paths
ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data" / "bids"
DATASET_ROOT: Path = ROOT / "data" / "dataset"
DATASET: Path = DATASET_ROOT / "naive_archive_2026-05-12_epo.fif"

# 1.2 Preprocessing
WINDOW_SIZE: float = 0.1

# 1.3. Model
TRAIN_CONFIG: TrainingConfig = TrainingConfig(
    n_splits=5,
    random_state=42,
    scoring="accuracy",
    shuffle=True
)



# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Loads the Dataset
    epochs = mne.read_epochs(DATASET, preload=True)

    # 2. Extract the envelop
    envelop_epochs = get_envelop(
        cast(mne.EpochsArray, epochs),
        window_s=WINDOW_SIZE,
    )

    # 3. Extract the features
    features = get_emg_features(
        envelop_epochs.get_data(),
        envelop_epochs.info["sfreq"],
        TIME_FEATURE_FUNCTIONS,
        FREQ_FEATURE_FUNCTIONS
    )
    print(
        f"Extracted {features.shape[1]} features from "
        f"{features.shape[0]} epochs\n"
    )

    # 4. Get all label names (we want to build one model per label)
    models = list(envelop_epochs.event_id.keys())
    print(f"Building the follwing models: {models}")

    # 5. Initializes the model architecture
    registry = ModelRegistry()
    registry.register(
        "dt",
        DecisionTreeFactory(
            random_state=42,
        ),
    )
    trainer = Trainer(
        registry=registry,
        target_builder=OneVsAllTargetBuilder(),
        evaluator=Evaluator(TRAIN_CONFIG),
    )

    # 5. Builds each model
    for model in models:
        print(f"|----- {model.title()} Model Training -----|")
        result = trainer.train(
            epochs=envelop_epochs,
            features=features,
            model_name="dt",
            target_name=model,
        )
        clf = result.estimator

        print(f"Decision tree: {model} vs no {model}")
        print("Mean accuracy:", result.mean_score)

        selected_features = prune_features(
            envelop_epochs,
            features,
            list(np.concat([TIME_FEATURE_FUNCTIONS, FREQ_FEATURE_FUNCTIONS], axis=0)),
            cast(DecisionTreeClassifier, clf),
            nr_to_keep = 7
        )
        print("Original:", features.shape)
        print("Selected:", selected_features.shape)

        result = trainer.train(
            epochs=envelop_epochs,
            features=selected_features,
            model_name="dt",
            target_name=model,
        )
        clf = result.estimator

        out_path = Path("data/models")
        os.makedirs(out_path, exist_ok=True)
        file_name = out_path / f"{model}_vs_all.joblib"
        joblib.dump(clf, file_name)

        print("Mean accuracy after pruning:", result.mean_score)
        print("Saved at:", file_name)
        print()
