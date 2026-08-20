import json, pandas as pd, os, numpy as np
from pathlib import Path

# === E06 full metrics ===
e06 = json.load(open('experiments/e06_hybrid_logreg/metrics.json'))
print('=== E06 FULL METRICS ===')
for split in ['train', 'validation', 'test', 'strict_test']:
    if split in e06:
        print(f'\n--- {split} ---')
        for k, v in e06[split].items():
            print(f'  {k}: {v}')

# === All experiments ===
print('\n=== ALL EXPERIMENT METRICS ===')
for d in sorted(os.listdir('experiments')):
    mp = os.path.join('experiments', d, 'metrics.json')
    if os.path.exists(mp):
        m = json.load(open(mp))
        if 'validation' in m:
            v = m['validation']
            t = m['test']
            print(f'{d}: ValAcc={v.get("accuracy", "N/A")}, ValF1={v.get("macro_f1", "N/A")}, TestAcc={t.get("accuracy", "N/A")}, TestF1={t.get("macro_f1", "N/A")}')
        elif 'val_macro_f1' in m:
            print(f'{d}: ValAcc={m.get("val_accuracy", "N/A")}, ValF1={m.get("val_macro_f1", "N/A")}, TestAcc={m.get("test_accuracy", "N/A")}, TestF1={m.get("test_macro_f1", "N/A")}')

# === Key manifests ===
print('\n=== EXECUTION LABEL BUILD ===')
elb = json.load(open('data/manifests/execution_label_build_manifest.json'))
for k, v in elb.items():
    print(f'  {k}: {v}')

print('\n=== FINAL LABEL DEFINITION ===')
fld = json.load(open('data/manifests/final_label_definition_v1.json'))
for k, v in fld.items():
    print(f'  {k}: {v}')

print('\n=== FINAL MODEL MANIFEST ===')
fmm = json.load(open('data/manifests/final_model_manifest_v1.json'))
for k, v in fmm.items():
    print(f'  {k}: {v}')

# === E06 predictions analysis ===
print('\n=== E06 PREDICTIONS ===')
preds = pd.read_parquet('experiments/e06_hybrid_logreg/final_test_predictions_v1.parquet')
print(f'  Columns: {list(preds.columns)}')
print(f'  Shape: {preds.shape}')
print(f'  True label distribution:\n{preds["true_label"].value_counts().sort_index()}')
print(f'  Predicted label distribution:\n{preds["predicted_label"].value_counts().sort_index()}')

# Recompute metrics from predictions to verify
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, f1_score)
y_true = preds['true_label']
y_pred = preds['predicted_label']
acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
p_per, r_per, f1_per, sup = precision_recall_fscore_support(y_true, y_pred, average=None)

print(f'\n=== RECOMPUTED FROM PREDICTIONS ===')
print(f'  Accuracy: {acc:.6f}')
print(f'  Balanced Accuracy: {bal_acc:.6f}')
print(f'  Macro Precision: {p_macro:.6f}')
print(f'  Macro Recall: {r_macro:.6f}')
print(f'  Macro F1: {f1_macro:.6f}')
print(f'  Weighted Precision: {p_w:.6f}')
print(f'  Weighted Recall: {r_w:.6f}')
print(f'  Weighted F1: {f1_w:.6f}')
for i in range(3):
    print(f'  Class {i}: P={p_per[i]:.6f} R={r_per[i]:.6f} F1={f1_per[i]:.6f} Support={sup[i]}')

# === Verify against stored metrics ===
stored = e06['test']
print(f'\n=== STORED vs RECOMPUTED ===')
print(f'  Accuracy: stored={stored["accuracy"]:.6f} recomputed={acc:.6f} match={np.isclose(stored["accuracy"], acc)}')
print(f'  Macro F1: stored={stored["macro_f1"]:.6f} recomputed={f1_macro:.6f} match={np.isclose(stored["macro_f1"], f1_macro)}')

# === Robustness ===
print('\n=== ROBUSTNESS SUMMARY ===')
rob = pd.read_csv('experiments/robustness_classical_v1/summary_metrics.csv')
print(rob.to_string())

print('\n=== ROBUSTNESS PER SEED ===')
rps = pd.read_csv('experiments/robustness_classical_v1/per_seed_metrics.csv')
print(rps.to_string())

# === Feature summary ===
print('\n=== FEATURE SUMMARY (first 10) ===')
fs = pd.read_csv('tables/feature_summary.csv')
print(fs.head(10).to_string())
print(f'  Total features: {len(fs)}')

# === Class distribution ===
print('\n=== CLASS DISTRIBUTION ===')
cd = pd.read_csv('tables/class_distribution.csv')
print(cd.to_string())

# === Normalization manifest ===
print('\n=== NORMALIZATION BUILD MANIFEST ===')
nm = json.load(open('data/manifests/normalization_build_manifest_v1.json'))
for k, v in nm.items():
    print(f'  {k}: {v}')
