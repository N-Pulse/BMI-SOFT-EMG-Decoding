import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from datetime import datetime

# def compute_scores(y_test, y_pred, plot_timeline=True, save_plots=True, plot_title="Model Predictions", save_dir="./results/plots"):
#     """
#     Enhanced compute_scores with timeline visualization and plot saving
    
#     Parameters:
#     - y_test: Ground truth labels
#     - y_pred: Predicted labels  
#     - plot_timeline: Whether to generate timeline plots
#     - save_plots: Whether to save plots to disk
#     - plot_title: Title for the plots
#     - save_dir: Directory to save plots
#     """
#     accuracy = accuracy_score(y_test, y_pred)
    
#     print(f"Accuracy: {100*accuracy:.2f} %")
#     print(f"Total samples: {len(y_test)}")
#     print(f"Correct predictions: {np.sum(y_test == y_pred)}")
#     print(f"Errors: {np.sum(y_test != y_pred)}")
#     print(f"Error rate: {100*np.sum(y_test != y_pred)/len(y_test):.2f} %")
    
#     if plot_timeline:
#         # Create save directory if it doesn't exist
#         if save_plots:
#             os.makedirs(save_dir, exist_ok=True)
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Generate the plots
#         fig = plot_timeline_comparison(y_test, y_pred, plot_title, save_plots, save_dir, timestamp)
        
#         if save_plots:
#             print(f"Plots saved to: {save_dir}")
    
#     return accuracy

def plot_timeline_comparison(y_test, y_pred, title, save_plots=False, save_dir=".", timestamp=""):
    """Plot y_test and y_pred along timeline and save if requested"""
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{title}\nOverall Accuracy: {100*accuracy_score(y_test, y_pred):.2f}%', 
                 fontsize=16, fontweight='bold')
    
    # Convert to numpy arrays for easier handling
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    indices = np.arange(len(y_test))
    
    # 1. Timeline comparison plot
    axes[0,0].plot(indices, y_test, 'o-', color='blue', label='Ground Truth', 
                   alpha=0.7, markersize=3, linewidth=1)
    axes[0,0].plot(indices, y_pred, 'x-', color='red', label='Predictions', 
                   alpha=0.7, markersize=3, linewidth=1)
    
    # Highlight errors
    errors = y_test != y_pred
    error_indices = indices[errors]
    if len(error_indices) > 0:
        axes[0,0].scatter(error_indices, y_test[error_indices], color='black', s=30, 
                         label='Errors', zorder=5, marker='s', alpha=0.8)
    
    axes[0,0].set_title('Timeline: Ground Truth vs Predictions', fontweight='bold')
    axes[0,0].set_xlabel('Sample Index (Time)')
    axes[0,0].set_ylabel('Class Label')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1], 
                cbar_kws={'label': 'Count'})
    axes[0,1].set_title('Confusion Matrix', fontweight='bold')
    axes[0,1].set_xlabel('Predicted Label')
    axes[0,1].set_ylabel('True Label')
    
    # 3. Error distribution over time
    # Calculate rolling accuracy (window size adaptive)
    window_size = max(1, min(50, len(y_test) // 20))
    if window_size > 1 and len(y_test) > window_size:
        rolling_accuracy = []
        for i in range(len(y_test) - window_size + 1):
            window_acc = np.mean(y_test[i:i+window_size] == y_pred[i:i+window_size])
            rolling_accuracy.append(window_acc)
        
        axes[1,0].plot(range(len(rolling_accuracy)), rolling_accuracy, 
                      color='green', linewidth=2, label=f'Rolling Accuracy (window={window_size})')
        axes[1,0].axhline(y=accuracy_score(y_test, y_pred), color='red', linestyle='--', 
                         label=f'Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}')
        axes[1,0].set_ylim(0, 1)
    else:
        # If not enough data for rolling, show binary correct/incorrect
        correct = y_test == y_pred
        axes[1,0].scatter(indices, correct, c=correct, cmap='RdYlGn', alpha=0.6)
        axes[1,0].set_yticks([0, 1])
        axes[1,0].set_yticklabels(['Wrong', 'Correct'])
    
    axes[1,0].set_title('Prediction Accuracy Over Time', fontweight='bold')
    axes[1,0].set_xlabel('Sample Index (Time)')
    axes[1,0].set_ylabel('Accuracy / Correctness')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    # 4. Class distribution comparison
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    true_counts = [np.sum(y_test == cls) for cls in unique_classes]
    pred_counts = [np.sum(y_pred == cls) for cls in unique_classes]
    
    x_pos = np.arange(len(unique_classes))
    width = 0.35
    
    axes[1,1].bar(x_pos - width/2, true_counts, width, label='True', alpha=0.7, color='blue')
    axes[1,1].bar(x_pos + width/2, pred_counts, width, label='Predicted', alpha=0.7, color='red')
    axes[1,1].set_title('Class Distribution: True vs Predicted', fontweight='bold')
    axes[1,1].set_xlabel('Class')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(unique_classes)
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        filename = f"prediction_timeline_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved as: {filepath}")
    
    plt.show()
    
    return fig

# Additional function for simple timeline-only plot
def plot_simple_timeline(y_test, y_pred, title="Prediction Timeline", save_plot=False, save_dir="./results/plots"):
    """Simple timeline plot focusing on the temporal alignment"""
    
    plt.figure(figsize=(15, 6))
    
    indices = np.arange(len(y_test))
    
    # Plot both lines
    plt.plot(indices, y_test, 'o-', color='blue', label='Ground Truth', 
             alpha=0.7, markersize=2, linewidth=0.5)
    plt.plot(indices, y_pred, 'x-', color='red', label='Predictions', 
             alpha=0.7, markersize=2, linewidth=0.5)
    
    # Highlight mismatches
    mismatches = y_test != y_pred
    mismatch_indices = indices[mismatches]
    if len(mismatch_indices) > 0:
        plt.scatter(mismatch_indices, y_test[mismatch_indices], 
                   color='black', s=20, label='Errors', zorder=5, alpha=0.8)
    
    accuracy = accuracy_score(y_test, y_pred)
    plt.title(f'{title}\nAccuracy: {100*accuracy:.2f}% | Errors: {np.sum(mismatches)}/{len(y_test)}', 
              fontweight='bold')
    plt.xlabel('Time (Sample Index)')
    plt.ylabel('Class Label')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_timeline_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Simple timeline saved as: {filepath}")
    
    plt.tight_layout()
    plt.show()

def compute_scores(y_test, y_pred, show=True, show_per_class=True):
    accuracy = accuracy_score(y_test, y_pred)
    
    # Main metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    if show:
        print(f"Accuracy: {100*accuracy:.2f} %")
        print(f"Precision: {100*precision:.2f} %")
        print(f"Recall: {100*recall:.2f} %")
        print(f"F1 Score: {100*f1:.2f} %")
        
        # Optional per-class display
        if show_per_class:
            classes = sorted(set(y_test) | set(y_pred))
            print("\nPer-class metrics:")
            for cls in classes:
                prec_cls = precision_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0]
                rec_cls = recall_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0]
                f1_cls = f1_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0]
                print(f"  Class {cls}: Precision={100*prec_cls:.1f}%, Recall={100*rec_cls:.1f}%, F1={100*f1_cls:.1f}%")

    return accuracy, precision, recall, f1

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