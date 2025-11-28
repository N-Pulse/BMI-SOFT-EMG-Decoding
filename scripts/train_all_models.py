import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import time
import json

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libML.data_load_and_label_for_training import load_emg_data
from libML.models import choose_model
from libML.evaluation import compute_scores, plot_cv_scores, plot_labels
from libML.export import save_best_params, save_model
from libML.preprocessing_new import segment_aux_windows_new, notch_filter, passband_filter
from libML.feature_engineering import extract_window_features

CONFIG = yaml.safe_load(open("config.yml"))
#TODO: Put all parameters in config file
# --- File/Directory Paths
PATHS = CONFIG.get('paths', {})
DATA_DIR = PATHS.get('raw_data_dir', './data/raw/')
PROCESSED_DATA_PATH = PATHS.get('processed_data_path', './data/processed/all_features.parquet')
MODEL_OUTPUT_DIR = PATHS.get('model_output_dir', './results/models/')
SCALER_OUTPUT_DIR = PATHS.get('scaler_output_dir', './results/scaler/')
BEST_PARAMS_OUTPUT_DIR = PATHS.get('best_params_output_dir', './results/params/')
FIG_OUTPUT_DIR = PATHS.get('fig_output_dir', './results/figures')
TIMING_OUTPUT_DIR = PATHS.get('timing_output_dir', './results/timing/')
SCALER_OUTPUT_DIR = PATHS.get('scaler_output_dir', './results/scaler/')

# --- Patient filters
FILTERS = CONFIG.get('filters', {})
SUBJECTS = FILTERS.get('subjects', "P005")
SESSIONS = FILTERS.get('sessions', "S002")
TASKS = FILTERS.get('tasks', "Default")
RUNS = FILTERS.get('runs', "001_eeg_up")

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
HYPERPARAMETER_SEARCH = MODELING.get('hyperparam_search', False)
NESTED_CV = MODELING.get('nested_cv', False)
ALL_PARAM_GRIDS = MODELING.get('param_grids', {})
NUM_TRIALS = MODELING.get('num_trials', 10)
N_SPLITS = MODELING.get('cv', 5)

MAP_DOF_NAME_TO_ID = {
        "thumb_flex_ext": "dof_1",
        "index_flex_ext": "dof_2",
        "middle_flex_ext": "dof_3",
        "ring_flex_ext": "dof_4",
        "pinky_flex_ext": "dof_5",
        "wrist_pro_sup": "dof_6",
        "wrist_flex_ext": "dof_7",
        "thumb_abd_add": "dof_8",
        # "strength": "dof_9",
        # "speed": "dof_10",
    }

def get_features(data_dict):
    # Get windows 
    X = data_dict["X"]
    y = data_dict["y"]
    windowed_df = segment_aux_windows_new(X, y)
    
    #breakpoint()
    # Apply preprocessing steps (filtering)
    preproc_df = notch_filter(windowed_df)
    preproc_df = passband_filter(preproc_df) 

    # Get features from windows
    # features_df contains all 6 channels with each feature --> 6*21 = 126 columns, named for example 0_AR4 --> channel id + _ + feature name
    # + 2 columns for label and window index so we have 128 columns    
    features_list = []
    for idx in range(len(preproc_df)):
        window_features = {}
        for ch in preproc_df.columns:
            if not(ch in [0,1,2,3,4,5]):
                continue
            signal_window = preproc_df.loc[idx, ch]
            #print(idx, ch)
            #print(signal_window)
            ch_features = extract_window_features(signal_window, fs=1000)
            
            # Prefix with channel name
            for feat_name, feat_val in ch_features.items():
                window_features[f"{ch}_{feat_name}"] = feat_val

        #preproc_df.keys()
        window_features['window_index'] = preproc_df.loc[idx, 'window_index']
        window_features['dof_1'] = preproc_df.loc[idx, 'dof_1_label']
        window_features['dof_2'] = preproc_df.loc[idx, 'dof_2_label']
        window_features['dof_3'] = preproc_df.loc[idx, 'dof_3_label']
        window_features['dof_4'] = preproc_df.loc[idx, 'dof_4_label']
        window_features['dof_5'] = preproc_df.loc[idx, 'dof_5_label']
        window_features['dof_6'] = preproc_df.loc[idx, 'dof_6_label']
        window_features['dof_7'] = preproc_df.loc[idx, 'dof_7_label']
        window_features['dof_8'] = preproc_df.loc[idx, 'dof_8_label']
        #window_features['dof_9'] = preproc_df.loc[idx, 'dof_9_label']
        #window_features['dof_10'] = preproc_df.loc[idx, 'dof_10_label']
        
        features_list.append(window_features)

    features_df = pd.DataFrame(features_list)
    #breakpoint()
    return features_df

def train_dof_model(dof, X_train, y_train, X_test, y_test):
    """Train and evaluate model for a single DOF"""
    print(f"Training model for {dof} ...")

    timing_data = {'dof': dof, 'model_type': MODEL_TYPE}
    start_time = time.time()

    y_dof_train = y_train[MAP_DOF_NAME_TO_ID[dof]]
    y_dof_test = y_test[MAP_DOF_NAME_TO_ID[dof]]
    
    if len(np.unique(y_dof_train)) == 1 or len(np.unique(y_dof_test)) == 1:
        print("Single label -> no classification")
        return dof, None, None, None, timing_data
    
    model = choose_model(MODEL_TYPE, ALL_HYPERPARAMS, RANDOM_STATE)
    best_params = None
    
    if HYPERPARAMETER_SEARCH:
        search_start = time.time()
        # Your existing hyperparameter search code with n_jobs=-1
        clf = GridSearchCV(model, ALL_PARAM_GRIDS[MODEL_TYPE], cv=KFold(n_splits=N_SPLITS), n_jobs=-1, return_train_score=True)
        clf.fit(X_train, y_dof_train)
        search_time = time.time() - search_start
        best_params = clf.best_params_
        timing_data['hyperparameter_search_time'] = search_time
        timing_data['best_score'] = clf.best_score_
        timing_data['best_params'] = best_params
        final_model = clf.best_estimator_
    else:
        training_start = time.time()
        final_model = model.fit(X_train, y_dof_train)
        training_time = time.time() - training_start
        best_params = final_model.get_params()
        timing_data['training_time'] = training_time
        timing_data['best_params'] = best_params
    
    total_time = time.time() - start_time
    timing_data['total_time'] = total_time

    # Evaluate
    y_pred = final_model.predict(X_test)
    test_scores = compute_scores(y_dof_test, y_pred)
    
    return dof, final_model, test_scores, best_params, timing_data

def save_timing_and_params(timing_data, best_params, model_type, dof, output_dir):
    """Save timing information and best parameters to files"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save timing info as CSV
    timing_filename = f"timing_{model_type}_{dof}_{timestamp}.csv"
    timing_filepath = os.path.join(output_dir, timing_filename)
    
    df = pd.DataFrame([timing_data])
    df.to_csv(timing_filepath, index=False)
    print(f"Timing info saved to {timing_filepath}")
    
    # Save best parameters as JSON
    params_filename = f"best_params_{model_type}_{dof}_{timestamp}.json"
    params_filepath = os.path.join(output_dir, params_filename)
    
    with open(params_filepath, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best parameters saved to {params_filepath}")
    
    # Also save combined info for easy analysis
    combined_data = timing_data.copy()
    combined_data['best_params'] = best_params
    combined_filename = f"combined_{model_type}_{dof}_{timestamp}.json"
    combined_filepath = os.path.join(output_dir, combined_filename)
    
    with open(combined_filepath, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)  # default=str handles non-serializable objects

def scale_features(processed_data, method='standard', save_path=None, return_dataframe=True):
    """
    Scale features and return scaled data with proper structure
    """
    # Handle input data
    if isinstance(processed_data, list):
        all_data = pd.concat(processed_data, ignore_index=True)
        was_list = True
    else:
        all_data = processed_data
        was_list = False
    
    # Identify columns
    label_cols = [col for col in all_data.columns if col.startswith('dof_')]
    meta_cols = [col for col in all_data.columns if col in ['window_index', 'label', 'subject_id', 'session_id']]
    feature_cols = [col for col in all_data.columns if col not in label_cols + meta_cols]
    
    X = all_data[feature_cols].values
    feature_names = feature_cols  # Keep track of feature names
    
    # Choose and fit scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        scaler_filename = f"scaler_{method}.pkl"
        scaler_filepath = os.path.join(save_path, scaler_filename)
        joblib.dump(scaler, scaler_filepath)
        
        # Also save feature names for reference
        feature_info = {
            'feature_names': feature_cols,
            'label_names': label_cols,
            'meta_names': meta_cols,
            'scaling_method': method
        }
        feature_info_path = os.path.join(save_path, f"feature_info_{method}.pkl")
        joblib.dump(feature_info, feature_info_path)
        print(f"Scaler and feature info saved to {save_path}")
    
    # Return appropriate format
    if return_dataframe:
        # Create new DataFrame with scaled features
        scaled_df = all_data.copy()
        scaled_df[feature_cols] = X_scaled
        
        if was_list:
            # Return list of DataFrames if input was list
            result = []
            start_idx = 0
            for df in processed_data:
                end_idx = start_idx + len(df)
                scaled_slice = scaled_df.iloc[start_idx:end_idx].reset_index(drop=True)
                result.append(scaled_slice)
                start_idx = end_idx
            return scaler, result
        else:
            return scaler, scaled_df
    else:
        return scaler, X_scaled

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
        (3.1) resample X_raw (emg timeseries) using the emg timestamps -> X_resampled)
        3.2) trigger vector y_triggers: same length as X_resampled, iterate throught timestamps and take the last triggers 
        3.3) map trigger to labels : y (in a dict for 8 DoFs) (+ add previous state of the arm pronation/neutral/supination)
        3.4) Window (only take windows with same labels) + filter (causal filters : bandpass + Notch) + feature extraction -> X_features (2D array), y_labels (dict of 1D arrays)
        3.5) Scale features, save scalers
        3.6) all to DataFrames : features_df = pd.DataFrame(X_features, columns=...), labels_df = pd.DataFrame(y_labels), session_df = pd.concat([features_df, labels_df], axis=1)
        3.7) session_df['subject_id'] = subject_id, session_df['session_id'] = session_id
        3.8) all_sessions_data.append(session_df)
        4) final_processed_df = pd.concat(all_sessions_data, ignore_index=True)
        5) Save features to avoid to compute it each time. to parquet ?
        '''
        # 1. Load data and 2. Map triggers into labels
        if os.path.exists(DATA_DIR):
            # Loads data from all the filters
            data_list = load_emg_data(DATA_DIR, subject=SUBJECTS, session=SESSIONS, task=TASKS, run=RUNS) # list of dict

        # 3. Preprocessing
        ## Resample at processing.sample_rate_hz
        ## Filter using bandpass + Notch
        ## Windowing -> feature extraction
        ## Feature standardization -> save generic scalers

        # Get features for all data available
        processed_data_list = []
        for run in data_list:
            # Get features for 1 run
            print("Getting features ...")
            processed_data = get_features(run)
            processed_data_list.append(processed_data)
            print("Got features!")

        # Save preprocessed data
        #save_features_to_disk(processed_data_list, PROCESSED_DATA_PATH)
    
    # TODO: Scale features and save scalers
    # 4. Scale features
    print("Scaling features...")
    scaler, scaled_data_list = scale_features(
        processed_data_list,
        method='standard',  # Choose: 'standard', 'robust', or 'minmax'
        save_path=SCALER_OUTPUT_DIR,
        return_dataframe=True
    )

    # 5. Train-test-split (using scaled data)
    all_data = pd.concat(scaled_data_list, ignore_index=True)
    
    # Separate features from labels
    label_cols = [col for col in all_data.columns if col.startswith('dof_')]
    feature_cols = [col for col in all_data.columns if col not in label_cols + ['window_index', 'label']]
    
    X = all_data[feature_cols].values
    y = {dof: all_data[dof].values for dof in label_cols}
    
    # Simple random split
    X_train, X_test = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_train = {dof: train_test_split(y[dof], test_size=TEST_SIZE, random_state=RANDOM_STATE)[0] for dof in label_cols}
    y_test = {dof: train_test_split(y[dof], test_size=TEST_SIZE, random_state=RANDOM_STATE)[1] for dof in label_cols}

    #breakpoint()

    # 5. Training models (loop for DoFs)
    trained_models = {}

    print("Training models for all DOFs in parallel...")
    start_time = time.time()
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_dof_model)(dof, X_train, y_train, X_test, y_test)
        for dof in DOF_LIST
    )

    # Process results
    trained_models = {}
    all_timing_data = {}
    all_best_params = {}

    for dof, model, scores, best_params, timing_data in results:
        if model is not None:
            trained_models[dof] = model
            all_timing_data[dof] = timing_data
            all_best_params[dof] = best_params
            
            # Save individual model results
            save_model(model, MODEL_TYPE, dof, MODEL_OUTPUT_DIR, scaler=None)
            save_timing_and_params(timing_data, best_params, MODEL_TYPE, dof, TIMING_OUTPUT_DIR)

    # Save summary across all DOFs
    summary_data = {
        'overall_timing': all_timing_data,
        'all_best_params': all_best_params,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    summary_filename = f"summary_all_dofs_{MODEL_TYPE}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_filepath = os.path.join(TIMING_OUTPUT_DIR, summary_filename)

    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    print(f"Overall summary saved to {summary_filepath}")
    end_time = time.time()
    print(f"Overall training performed in {(end_time-start_time):.2f} seconds.")

    # for dof in DOF_LIST:
    #     # 5.1. Select corresponding labels
    #     print("\n")
    #     print(f"Training model for {dof} ...")
    #     # breakpoint()
    #     y_dof_train = y_train[MAP_DOF_NAME_TO_ID[dof]] # labels corresponding to DoF
    #     y_dof_test = y_test[MAP_DOF_NAME_TO_ID[dof]]

    #     if len(np.unique(y_dof_train))==1 or len(np.unique(y_dof_test))==1:
    #         print("Single label -> no classification")
    #         continue

    #     # 5.2. Model
    #     model = choose_model(MODEL_TYPE, ALL_HYPERPARAMS, RANDOM_STATE)

        
    #     if HYPERPARAMETER_SEARCH:
    #         # 5.3. Hyperparameter search
    #         print(f"Hyperparameter search ...")
    #         param_grid = ALL_PARAM_GRIDS.get(MODEL_TYPE, None)
    #         kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    #         if NESTED_CV:
    #             print(f"Nested CV ...")
    #             scores = []
    #             # print("Without parallelization, this may take a while...")
    #             # start_time = time.time()
    #             # for i in range(NUM_TRIALS):
    #             #     clf = GridSearchCV(model, param_grid, cv=kfold)
    #             #     score = cross_val_score(clf, X=X_train, y=y_dof_train, cv=kfold)
    #             #     scores[i] = score.mean()
    #             #     best_params = clf.best_params_
    #             #     print(f"Trial {i+1}/{NUM_TRIALS}, Best Params: {best_params}, Score: {score.mean():.4f}")
    #             # end_time = time.time()
    #             # print(f"Hyperparameter search (without parallelization)complete in {end_time - start_time:.2f} seconds.")
    #             # plot_cv_scores(scores)

    #             scores = []
    #             start_time = time.time()
    #             print("With parallelization ...")
    #             for i in range(NUM_TRIALS):
    #                 clf = GridSearchCV(model, param_grid, cv=kfold, n_jobs=-1)
    #                 score = cross_val_score(clf, X=X_train, y=y_dof_train, cv=kfold)
    #                 scores[i] = score.mean()
    #                 best_params = clf.best_params_
    #                 print(f"Trial {i+1}/{NUM_TRIALS}, Best Params: {best_params}, Score: {score.mean():.4f}")
    #             end_time = time.time()
    #             print(f"Hyperparameter search (without parallelization)complete in {end_time - start_time:.2f} seconds.")
    #             plot_cv_scores(scores)

    #         else:
    #             print(f"CV ...")
    #             clf = GridSearchCV(
    #                 estimator=model,
    #                 param_grid=param_grid,
    #                 cv=kfold,
    #             )
    #             clf.fit(X_train, y_dof_train)
    #             best_params = clf.best_params_
    #             best_score = clf.best_score_
    #             final_model = clf.best_estimator_
    #             save_best_params(best_params, MODEL_TYPE, dof, BEST_PARAMS_OUTPUT_DIR)
    #     else:
    #         # 5.4. Train model with best params
    #         print(f"No hyperparameter search, direct training ...")
    #         final_model = model
    #         best_params = final_model.get_params()
    #         print(f"best params: {best_params}")
    #         final_model.fit(X_train, y_dof_train)
    # TODO: train time, inference time --> load a model and input a random window
    #     # 5.5. Evaluate
    #     y_pred = final_model.predict(X_test)
    #     test_scores = compute_scores(y_dof_test, y_pred)

    #     save_plot_path = os.path.join(FIG_OUTPUT_DIR, f"{MODEL_TYPE}/plot_labels_{dof}.png")
    #     # plot_labels(y_dof_test, y_pred, show=False, save_path=save_plot_path)

    #     # 5.6. Save generic model
    #     save_model(final_model, MODEL_TYPE, dof, MODEL_OUTPUT_DIR, scaler=None)
    #     trained_models[dof] = final_model


if __name__ == "__main__":
    main()

