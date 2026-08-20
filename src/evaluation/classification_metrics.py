import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)

def evaluate_predictions(y_true, y_pred, labels=[0, 1, 2]):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    }
    
    # Per class metrics
    precisions = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    
    metrics["per_class_precision"] = {str(lbl): float(p) for lbl, p in zip(labels, precisions)}
    metrics["per_class_recall"] = {str(lbl): float(r) for lbl, r in zip(labels, recalls)}
    metrics["per_class_f1"] = {str(lbl): float(f) for lbl, f in zip(labels, f1s)}
    
    return metrics

def save_confusion_matrix_plot(y_true, y_pred, filepath, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['LOW (0)', 'MEDIUM (1)', 'HIGH (2)'])
    ax.set_yticklabels(['LOW (0)', 'MEDIUM (1)', 'HIGH (2)'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return cm.tolist()
