"""Evaluation helpers for a trained TCNN: test metrics, learning curves, and saved results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from bmiemg.models.evaluation import evaluate_classification


def save_loss_curve(history, path):
    train_loss = [train.loss for train, val in history]   # training loss of each epoch
    val_loss = [val.loss for train, val in history]       # validation loss of each epoch
    epochs = range(1, len(history) + 1)                   # 1, 2, …, N (range end is excluded)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(path)
    plt.close()

def save_accuracy_curve(history, path):
    train_acc = [100 * train.accuracy for train, val in history]   # training accuracy of each epoch (%)
    val_acc = [100 * val.accuracy for train, val in history]       # validation accuracy of each epoch (%)
    epochs = range(1, len(history) + 1)                            # 1, 2, …, N

    plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig(path)
    plt.close()

def predict_labels(model, loader, device):
    model.eval()                                    # evaluation mode

    true_gestures = []                              # ground-truth gestures
    predicted_gestures = []                         # gestures predicted by the network

    with torch.no_grad():                           # no gradients: weights stay unchanged
        for signals, targets in loader:             # one batch of windows + their true gestures
            signals = signals.to(device)            # move windows to the model's device
            scores = model(signals)                 # 6 scores per window
            predictions = scores.argmax(dim=1)      # gesture with the highest score
            predictions = predictions.cpu()         # bring the result back to the CPU

            predicted_gestures.extend(predictions.tolist())   # append predicted gestures
            true_gestures.extend(targets.tolist())            # append true gestures

    return np.array(true_gestures), np.array(predicted_gestures)


def compute_test_metrics(model, test_loader, device, class_labels):
    true_gestures, predicted_gestures = predict_labels(model, test_loader, device)   # class indices
    true_names = [class_labels[i] for i in true_gestures]            # 1 → "fist", 5 → "pinch", …
    predicted_names = [class_labels[i] for i in predicted_gestures]  # same for predictions

    return evaluate_classification(true_names, predicted_names)      # all metrics at once


def save_confusion_matrix_plot(evaluation, path):
    counts = evaluation.confusion_matrix                     # raw counts (rows = true gestures)
    row_totals = counts.sum(axis=1, keepdims=True)           # number of windows per true gesture
    percentages = 100 * counts / np.maximum(row_totals, 1)   # each row now sums to 100 %

    display = ConfusionMatrixDisplay(
        confusion_matrix=percentages,                        # plot percentages instead of counts
        display_labels=evaluation.labels,                    # gesture names
    )
    display.plot(cmap="Blues", xticks_rotation=45, values_format=".1f")  # 1 decimal, e.g. 87.5
    plt.title("Confusion matrix (test, % of true gesture)")
    plt.savefig(path, bbox_inches="tight")
    plt.close()                                      # free memory


def save_results(evaluation, history, users, epochs, window_ms):
    folder = Path(f"scripts/cnn/results/{users}user_{epochs}epochs_{window_ms:g}ms_withRandomRotation")
    folder.mkdir(parents=True, exist_ok=True)                                           # create the folder if needed

    save_loss_curve(history, folder / "loss_curve.png")    
    save_accuracy_curve(history, folder / "accuracy_curve.png")      # accuracy curves# loss curves
    save_confusion_matrix_plot(evaluation, folder / "confusion_matrix.png")  # confusion matrix image

    with open(folder / "metrics.json", "w") as file:                         # accuracy, F1, report, …
        json.dump(evaluation.as_dict(), file, indent=2)

    matrix = pd.DataFrame(
        evaluation.confusion_matrix,
        index=evaluation.labels,      # rows = true gestures
        columns=evaluation.labels,    # columns = predicted gestures
    )
    matrix.to_csv(folder / "confusion_matrix.csv")                           # confusion matrix values
    
