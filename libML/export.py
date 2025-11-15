import os
import json
import joblib
import numpy as np

def save_best_params(params, dof_name, dir):

    for key, value in params.items():
        if isinstance(value, np.bool_):
            params[key] = bool(value)
        elif isinstance(value, np.integer):
            params[key] = int(value)
        elif isinstance(value, np.floating):
            params[key] = float(value)
            
    params_path = os.path.join(dir, f"best_params_{dof_name}.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Saved best params to {params_path}")


def save_model(model, scaler, dof_name, output_dir):
    """
    Saves the trained model and its associated scaler to disk.

    Args:
        model (object): The trained scikit-learn model (e.g., LDA, SVM).
        scaler (StandardScaler): The scikit-learn StandardScaler
                                 that was fit on the training data.
        dof_name (str): The name of the degree of freedom (e.g., "wrist_flex_ext").
        output_dir (str): The directory path to save the files to.
    """
    try:
        # 1. Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Define file paths
        model_path = os.path.join(output_dir, f"model_{dof_name}.joblib")
        scaler_path = os.path.join(output_dir, f"scaler_{dof_name}.joblib")

        # 3. Save the Model
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")

        # 4. Save the Scaler
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to: {scaler_path}")

    except Exception as e:
        print(f"Error saving model or scaler for {dof_name}: {e}")