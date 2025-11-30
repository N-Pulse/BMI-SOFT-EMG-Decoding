import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from matplotlib import pyplot as plt

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

    