import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from matplotlib import pyplot as plt
import numpy as np

def compute_scores(y_test, y_pred, show=True):
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get unique classes
    classes = sorted(set(y_test) | set(y_pred))
    
    # Compute per-class metrics
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    for cls in classes:
        precision_dict[cls] = precision_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0]
        recall_dict[cls] = recall_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0]
        f1_dict[cls] = f1_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0]
    
    if show:
        print(f"Accuracy: {100*accuracy:.2f} %")
        print(f"Precision per class: {precision_dict}")
        print(f"Recall per class: {recall_dict}")
        print(f"F1 Score per class: {f1_dict}")

    return accuracy, precision_dict, recall_dict, f1_dict

def plot_cv_scores(scores):
    plt.figure()
    plt.plot(scores, color='r')
    plt.ylabel("Score")
    plt.title("Cross Validation Scores")
    plt.show()

def plot_labels(y_test, y_pred, show=True, save_path=None):
    plt.figure()
    plt.plot(y_test, label='True Labels', alpha=0.7)
    plt.plot(y_pred, label='Predicted Labels', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("True vs Predicted Labels")
    plt.legend()

    if show:
        plt.show()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path)
        plt.close()

def evaluate_overall_system_performance_detailed(trained_models, X_test, y_test, dof_list, MAP_DOF_NAME_TO_ID):
    """
    Detailed overall system performance with error analysis
    """
    if not trained_models:
        return {}
    
    # Get predictions
    all_predictions = {}
    for dof in dof_list:
        if dof in trained_models and trained_models[dof] is not None:
            all_predictions[dof] = trained_models[dof].predict(X_test)
    
    if not all_predictions:
        return {}
    
    n_samples = len(X_test)
    
    # Calculate per-sample correctness
    sample_correct = np.ones(n_samples, dtype=bool)
    error_breakdown = {dof: 0 for dof in all_predictions.keys()}
    
    for dof, predictions in all_predictions.items():
        correct_dof = (predictions == y_test[MAP_DOF_NAME_TO_ID[dof]])
        sample_correct &= correct_dof
        
        # Count errors per DOF
        error_breakdown[dof] = np.sum(~correct_dof)
    
    n_correct = np.sum(sample_correct)
    overall_accuracy = n_correct / n_samples
    
    # Error analysis
    total_errors = n_samples - n_correct
    error_per_dof = {dof: count/total_errors if total_errors > 0 else 0 
                    for dof, count in error_breakdown.items()}
    
    results = {
        'overall_accuracy': overall_accuracy,
        'n_correct_samples': int(n_correct),
        'n_total_samples': n_samples,
        'correct_percentage': overall_accuracy * 100,
        'total_errors': int(total_errors),
        'errors_per_dof': error_breakdown,
        'error_ratio_per_dof': error_per_dof
    }
    
    # Which DOFs are most problematic?
    if total_errors > 0:
        most_problematic_dof = max(error_breakdown.items(), key=lambda x: x[1])
        results['most_problematic_dof'] = most_problematic_dof[0]
        results['max_errors'] = most_problematic_dof[1]
    
    return results