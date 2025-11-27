import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from matplotlib import pyplot as plt

def compute_scores(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print(f"Accuracy: {accuracy}")
    return accuracy#, precision, recall, f1

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

    