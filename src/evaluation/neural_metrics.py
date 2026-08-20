import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

def compute_neural_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'weighted_f1': w_f1,
        'class_0_precision': p_c[0] if len(p_c) > 0 else 0.0,
        'class_1_precision': p_c[1] if len(p_c) > 1 else 0.0,
        'class_2_precision': p_c[2] if len(p_c) > 2 else 0.0,
        'class_0_recall': r_c[0] if len(r_c) > 0 else 0.0,
        'class_1_recall': r_c[1] if len(r_c) > 1 else 0.0,
        'class_2_recall': r_c[2] if len(r_c) > 2 else 0.0,
        'class_0_f1': f1_c[0] if len(f1_c) > 0 else 0.0,
        'class_1_f1': f1_c[1] if len(f1_c) > 1 else 0.0,
        'class_2_f1': f1_c[2] if len(f1_c) > 2 else 0.0,
    }
