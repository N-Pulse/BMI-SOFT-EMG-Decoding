import os
import yaml
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from libML import data_load_and_label_for_training
from data_load_and_label_for_training import load_emg_data

from libML.models import choose_model
from libML.evaluation import compute_scores, plot_cv_scores
from libML.export import save_best_params, save_model

CONFIG = yaml.safe_load(open("config.yml"))

# --- File/Directory Paths
PATHS = CONFIG.get('paths', {})
DATA_DIR = PATHS.get('raw_data_dir', './data/raw/')
PROCESSED_DATA_PATH = PATHS.get('processed_data_path', './data/processed/all_features.parquet')
MODEL_OUTPUT_DIR = PATHS.get('model_output_dir', './results/models/')
SCALER_OUTPUT_DIR = PATHS.get('scaler_output_dir', './results/scaler/')
BEST_PARAMS_OUTPUT_DIR = PATHS.get('best_params_output_dir', './results/params/')
FIG_OUTPUT_DIR = PATHS.get('fig_output_dir', './results/figures')

# --- Windowing & Feature Parameters
PROC = CONFIG.get('processing', {})
SAMPLE_RATE_HZ = PROC.get('sample_rate_hz', 2000)
WINDOW_SIZE_MS = PROC.get('window_size_ms', 200)
WINDOW_STEP_MS = PROC.get('window_step_ms', 50)
PREPROCESSING = PROC.get('preprocessing', True)

WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE_HZ * (WINDOW_SIZE_MS / 1000.0))
WINDOW_STEP_SAMPLES = int(SAMPLE_RATE_HZ * (WINDOW_STEP_MS / 1000.0))

# --- Model & Training Parameters
MODELING = CONFIG.get('modeling', {})
DOF_LIST = MODELING.get('dof_list', []) # Default to empty list
MODEL_TYPE = MODELING.get('model_type', 'LDA')
ALL_HYPERPARAMS = MODELING.get('hyperparams', {})

RANDOM_STATE = MODELING.get('random_state', 42)
TEST_SIZE = MODELING.get('test_size', 0.2)

# Grid search
HYPERPARAMETER_SEARCH = MODELING.get('hyperparam_search', True)
NESTED_CV = MODELING.get('nested_cv', False)
ALL_PARAM_GRIDS = MODELING.get('param_grids', {})
NUM_TRIALS = MODELING.get('num_trials', 10)
N_SPLITS = MODELING.get('n_splits', 4)


def main():

    # 1-3. Load data, preprocess, extract features
    if PREPROCESSING == False and os.path.exists(PROCESSED_DATA_PATH):
        # 1-3. Load preprocessed data
        processed_data = load_feat_from_disk(PROCESSED_DATA_PATH)

    else:
        '''
        TODO
        1) loop for subjects and for sessions
        2) load emg + triggers
        3) empty all_sessions_data. For each emg file : 
        3.1) resample X_raw (emg timeseries) using the emg timestamps -> X_resampled  
        3.2) trigger vector y_triggers: same length as X_resampled, iterate throught timestamps and take the last triggers 
        3.3) map trigger to labels : y (in a dict for 8 DoFs) 
        3.4) Filter : bandpass + Notch
        3.5) Window + feature extraction -> X_features (2D array), y_labels (dict of 1D arrays)
        3.6) all to DataFrames : features_df = pd.DataFrame(X_features, columns=...), labels_df = pd.DataFrame(y_labels), session_df = pd.concat([features_df, labels_df], axis=1)
        3.7) session_df['subject_id'] = subject_id, session_df['session_id'] = session_id
        3.8) all_sessions_data.append(session_df)
        4) final_processed_df = pd.concat(all_sessions_data, ignore_index=True)
        5) Save features to avoid to compute it each time. to parquet ?
        '''

        # 1. Load data and 2. Map triggers into labels
        if os.path.exists(DATA_DIR):
            #emg_raw_data = load_emg_data(DATA_DIR)
            # load_emg_data loads data for one specific run/task/session/subject combination, execute in a loop to get data from all subjects and sessions available (default: subject="P005", session="S002", task="Default", run="001_eeg_up")
            X_raw, y_raw_dict = load_emg_data(DATA_DIR)

        # 3. Preprocessing
        # TODO:
        ## Resample at processing.sample_rate_hz
        ## Filter using bandpass + Notch
        ## Windowing -> feature extraction
        ## Feature standardization -> save generic scalers

        # Save preprocessed data
        save_features_to_disk(processed_data, PROCESSED_DATA_PATH)

    # 4. Train-test-split (by subject or session)
    X_train, y_train, X_test, y_test = ...

    # 5. Training models (loop for DoFs)
    trained_models = {}
    for dof in DOF_LIST:

        # 5.1. Select corresponding labels
        y_dof = y[dof]
        X_dof = X

        # 5.2. Model
        model = choose_model(MODEL_TYPE, ALL_HYPERPARAMS, RANDOM_STATE)

        
        if HYPERPARAMETER_SEARCH:
            # 5.3. Hyperparameter search
            param_grid = ALL_PARAM_GRIDS.get(MODEL_TYPE, None)
            kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

            if NESTED_CV:
                scores = []
                for i in range(NUM_TRIALS):
                    clf = GridSearchCV(model, param_grid, cv=kfold)
                    score = cross_val_score(clf, X=X_train, y=y_train, cv=kfold)
                    scores[i] = score.mean()
                    best_params = clf.best_params_
                plot_cv_scores(scores)
            else:
                clf = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=kfold,
                    n_jobs=-1,
                    verbose=-1
                )
                clf.fit(X_train, y_train)
                best_params = clf.best_params_
                best_score = clf.best_score_
                final_model = clf.best_estimator_
                save_best_params(best_params, dof, BEST_PARAMS_OUTPUT_DIR)
        else:
            # 5.4. Train model with best params
            final_model.fit(X_train, y_train)

        # 5.5. Evaluate
        y_pred = model.predict(X_test)
        test_scores = compute_scores(y_test, y_pred)

        # 5.6. Save generic model
        save_model(model, dof, best_params, MODEL_OUTPUT_DIR)
        trained_models[dof] = model


if __name__ == "__main__":
    main()

