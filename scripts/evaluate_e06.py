import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import yaml
import time
import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    balanced_accuracy_score, matthews_corrcoef, 
    log_loss, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.models.hybrid_classical import HybridBaseline

def evaluate_e06():
    print("Starting E06 Hybrid Logistic Regression Evaluation...")
    
    dataset_path = 'data/processed/model_ready_hybrid_v1.parquet'
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset missing at {dataset_path}")
        sys.exit(1)
        
    model_path = 'experiments/e06_hybrid_logreg/model.joblib'
    if not os.path.exists(model_path):
        print(f"ERROR: Model missing at {model_path}")
        sys.exit(1)
        
    print("Loading dataset...")
    df = pd.read_parquet(dataset_path)
    
    prov_path = 'data/processed/model_provenance_v1.parquet'
    if not os.path.exists(prov_path):
        print(f"ERROR: Provenance dataset missing at {prov_path}")
        sys.exit(1)
    df_prov = pd.read_parquet(prov_path)
    
    df = df.merge(df_prov[['sample_id', 'repository']], on='sample_id', how='left')
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'test']
    
    train_repos = set(train_df['repository'])
    val_repos = set(val_df['repository'])
    test_repos = set(test_df['repository'])
    
    int_train_val = train_repos.intersection(val_repos)
    int_train_test = train_repos.intersection(test_repos)
    int_val_test = val_repos.intersection(test_repos)
    
    print(f"\nRepository Disjointness Check:")
    print(f"  Train n Val: {len(int_train_val)}")
    print(f"  Train n Test: {len(int_train_test)}")
    print(f"  Val n Test: {len(int_val_test)}")
    
    if len(int_train_val) > 0 or len(int_train_test) > 0 or len(int_val_test) > 0:
        print("ERROR: Repository leakage detected! Halting evaluation.")
        sys.exit(1)
        
    print("\nClass Distribution:")
    for split_name, split_df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
        counts = split_df['final_label'].value_counts().sort_index()
        total = len(split_df)
        print(f"  {split_name}:")
        for k, v in counts.items():
            print(f"    Class {k}: {v} ({v/total*100:.1f}%)")
            
    print("\nLoading model...")
    model = HybridBaseline.load(model_path)
    
    text_col = "normalized_workflow_text"
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", text_col] + diagnostic_features]
    
    X_test_text = test_df[text_col]
    X_test_struct = test_df[struct_cols]
    y_test = test_df['final_label'].values
    
    print("\nEvaluating model on TEST set...")
    
    preds = model.predict(X_test_text, X_test_struct)
    
    try:
        X_hybrid = model.transform(X_test_text, X_test_struct)
        probs = model.model.predict_proba(X_hybrid)
    except Exception as e:
        print(f"Warning: Probability extraction failed: {e}")
        probs = None
        
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
    
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_test, preds, average=None)
    
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "macro_avg": {"precision": p_mac, "recall": r_mac, "f1-score": f1_mac},
        "weighted_avg": {"precision": p_wt, "recall": r_wt, "f1-score": f1_wt},
        "per_class": {
            "LOW": {"precision": p_per[0], "recall": r_per[0], "f1-score": f1_per[0], "support": int(sup_per[0])},
            "MEDIUM": {"precision": p_per[1], "recall": r_per[1], "f1-score": f1_per[1], "support": int(sup_per[1])},
            "HIGH": {"precision": p_per[2], "recall": r_per[2], "f1-score": f1_per[2], "support": int(sup_per[2])}
        }
    }
    
    if probs is not None:
        metrics["log_loss"] = log_loss(y_test, probs)
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        except:
            metrics["roc_auc_ovr_macro"] = None
            
    out_dir = Path("experiments/e06_hybrid_logreg")
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    class_rep = classification_report(y_test, preds, target_names=["LOW", "MEDIUM", "HIGH"], output_dict=True)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(class_rep, f, indent=4)
        
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=["LOW", "MEDIUM", "HIGH"], columns=["LOW", "MEDIUM", "HIGH"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: E06 Hybrid LogReg (Test Set)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    
    meta = {
        "dataset_path": dataset_path,
        "model_path": model_path,
        "split_method": "repository_stratified",
        "random_seed": 42,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "train_repositories": len(train_repos),
        "validation_repositories": len(val_repos),
        "test_repositories": len(test_repos),
        "feature_count_structural": len(struct_cols)
    }
    
    with open(out_dir / "evaluation_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
        
    print("\n--- TEST METRICS ---")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Macro Precision:   {p_mac:.4f}")
    print(f"Macro Recall:      {r_mac:.4f}")
    print(f"Macro F1:          {f1_mac:.4f}")
    print(f"Weighted F1:       {f1_wt:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"MCC:               {mcc:.4f}")
    if probs is not None:
        print(f"Log Loss:          {metrics['log_loss']:.4f}")
        print(f"ROC-AUC (OVR):     {metrics['roc_auc_ovr_macro']:.4f}")
        
    print("\nEvaluation successfully completed. Artifacts saved to experiments/e06_hybrid_logreg/")

if __name__ == '__main__':
    evaluate_e06()
