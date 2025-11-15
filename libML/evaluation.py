from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from matplotlib import pyplot as plt

def compute_scores(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    return accuracy, precision, recall, f1

def plot_cv_scores(scores):
    plt.figure()
    plt.plot(scores, color='r')
    plt.ylabel("Score")
    plt.title("Cross Validation Scores")
    plt.show()