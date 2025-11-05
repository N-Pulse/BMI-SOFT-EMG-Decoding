import joblib
from libML import data_loading, preprocessing, feature_extraction, model_utils, evaluation

def train_single_dof(dof_name, params):
    """
    Train a decoding model for one DoF using EMG data.
    """
    # 1. Load data
    X_raw, y = data_loading.load_emg_data(dof_name)

    # 2. Preprocess
    X_preproc = preprocessing.apply_preprocessing(X_raw)

    # 3. Feature extraction
    X_feat = feature_extraction.extract_features_from_dataset(X_preproc)

    # 4. Train model
    model = model_utils.train_model(X_feat, y, params)

    # 5. Evaluate
    metrics = evaluation.evaluate_model(model, X_feat, y)
    print(f"{dof_name}: {metrics}")

    # 6. Save
    joblib.dump(model, f"models/{dof_name}.pkl")
