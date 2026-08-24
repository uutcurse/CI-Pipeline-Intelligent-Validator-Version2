import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    balanced_accuracy_score, matthews_corrcoef, 
    log_loss, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.hybrid_classical import HybridBaseline

def evaluate_model(y_true, y_pred, y_probs=None, multi_class="ovr"):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    ll = None
    auc = None
    if y_probs is not None:
        try:
            ll = log_loss(y_true, y_probs)
            auc = roc_auc_score(y_true, y_probs, multi_class=multi_class, average="macro")
        except:
            pass
            
    return {
        "Accuracy": acc,
        "Macro F1": f1_mac,
        "Weighted F1": f1_wt,
        "Balanced Accuracy": bal_acc,
        "MCC": mcc,
        "Log Loss": ll,
        "ROC-AUC": auc,
        "Per-Class": {
            0: {"Precision": p_per[0], "Recall": r_per[0], "F1": f1_per[0], "Support": sup_per[0]},
            1: {"Precision": p_per[1], "Recall": r_per[1], "F1": f1_per[1], "Support": sup_per[1]},
            2: {"Precision": p_per[2], "Recall": r_per[2], "F1": f1_per[2], "Support": sup_per[2]}
        },
        "y_pred": y_pred
    }

def plot_cm(y_true, y_pred, filename, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["LOW", "MEDIUM", "HIGH"], columns=["LOW", "MEDIUM", "HIGH"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("Loading data...")
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    df = df.merge(df_prov[['sample_id', 'repository']], on='sample_id', how='left')
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'test']
    
    train_repos = set(train_df['repository'])
    val_repos = set(val_df['repository'])
    test_repos = set(test_df['repository'])
    
    i1 = train_repos.intersection(val_repos)
    i2 = train_repos.intersection(test_repos)
    i3 = val_repos.intersection(test_repos)
    print(f"Repository Disjointness Check:")
    print(f"Train n Val: {len(i1)}")
    print(f"Train n Test: {len(i2)}")
    print(f"Val n Test: {len(i3)}")
    if len(i1)>0 or len(i2)>0 or len(i3)>0:
        print("ERROR: Repository Leakage detected.")
        sys.exit(1)
        
    text_col = "normalized_workflow_text"
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", text_col] + diagnostic_features]
    
    X_train_text, X_train_struct, y_train = train_df[text_col], train_df[struct_cols], train_df['final_label'].values
    X_val_text, X_val_struct, y_val = val_df[text_col], val_df[struct_cols], val_df['final_label'].values
    X_test_text, X_test_struct, y_test = test_df[text_col], test_df[struct_cols], test_df['final_label'].values
    
    # Text Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, strip_accents='unicode')
    X_train_text_v = vectorizer.fit_transform(X_train_text)
    X_val_text_v = vectorizer.transform(X_val_text)
    X_test_text_v = vectorizer.transform(X_test_text)
    
    # Struct Scaler
    scaler = StandardScaler()
    X_train_struct_s = scaler.fit_transform(X_train_struct)
    X_val_struct_s = scaler.transform(X_val_struct)
    X_test_struct_s = scaler.transform(X_test_struct)
    
    results = {}
    
    # B0: Majority
    print("Evaluating B0: Majority Baseline")
    maj_class = pd.Series(y_train).mode()[0]
    preds_b0 = np.full_like(y_test, fill_value=maj_class)
    # create dummy probs for LogLoss (0 for non-majority, 1 for majority)
    probs_b0 = np.zeros((len(y_test), 3))
    probs_b0[:, maj_class] = 1.0
    results["B0 (Majority)"] = evaluate_model(y_test, preds_b0, probs_b0)
    
    # B1: LogReg Structural
    print("Evaluating B1: LogReg Structural")
    m_b1 = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42)
    m_b1.fit(X_train_struct_s, y_train)
    results["B1 (LogReg Struct)"] = evaluate_model(y_test, m_b1.predict(X_test_struct_s), m_b1.predict_proba(X_test_struct_s))
    
    # B2: LogReg Text
    print("Evaluating B2: LogReg Text")
    m_b2 = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42)
    m_b2.fit(X_train_text_v, y_train)
    results["B2 (LogReg Text)"] = evaluate_model(y_test, m_b2.predict(X_test_text_v), m_b2.predict_proba(X_test_text_v))
    
    # B3: RandomForest Structural
    print("Evaluating B3: Random Forest Structural")
    m_b3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    m_b3.fit(X_train_struct_s, y_train)
    results["B3 (RF Struct)"] = evaluate_model(y_test, m_b3.predict(X_test_struct_s), m_b3.predict_proba(X_test_struct_s))
    
    # B4: XGBoost Structural
    print("Evaluating B4: XGBoost Structural")
    m_b4 = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    m_b4.fit(X_train_struct_s, y_train)
    results["B4 (XGB Struct)"] = evaluate_model(y_test, m_b4.predict(X_test_struct_s), m_b4.predict_proba(X_test_struct_s))
    
    # B5: Existing E06 Hybrid
    print("Evaluating B5: E06 Hybrid LogReg")
    e06_model = HybridBaseline.load("experiments/e06_hybrid_logreg/model.joblib")
    preds_e06 = e06_model.predict(X_test_text, X_test_struct)
    
    probs_e06 = None
    try:
        X_hybrid = e06_model.transform(X_test_text, X_test_struct)
        probs_e06 = e06_model.model.predict_proba(X_hybrid)
    except Exception as e:
        pass
    results["B5 (E06 Hybrid)"] = evaluate_model(y_test, preds_e06, probs_e06)
    
    e06_mac = results["B5 (E06 Hybrid)"]["Macro F1"]
    if abs(e06_mac - 0.4972) > 0.001:
        print(f"ERROR: E06 Macro F1 mismatch! Expected ~0.4972, got {e06_mac}")
        sys.exit(1)
        
    print("Writing reports...")
    out_dir = Path("experiments/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    
    plot_cm(y_test, results["B1 (LogReg Struct)"]["y_pred"], cm_dir / "logreg_structural.png", "B1: LogReg Structural")
    plot_cm(y_test, results["B2 (LogReg Text)"]["y_pred"], cm_dir / "logreg_text.png", "B2: LogReg Text")
    plot_cm(y_test, results["B3 (RF Struct)"]["y_pred"], cm_dir / "random_forest.png", "B3: Random Forest")
    plot_cm(y_test, results["B4 (XGB Struct)"]["y_pred"], cm_dir / "xgboost.png", "B4: XGBoost")
    plot_cm(y_test, results["B5 (E06 Hybrid)"]["y_pred"], cm_dir / "e06_hybrid.png", "B5: E06 Hybrid")
    
    # Overall Results CSV
    res_records = []
    for m, metrics in results.items():
        res_records.append({
            "Model": m,
            "Accuracy": metrics["Accuracy"],
            "Macro F1": metrics["Macro F1"],
            "Weighted F1": metrics["Weighted F1"],
            "Balanced Accuracy": metrics["Balanced Accuracy"],
            "MCC": metrics["MCC"],
            "Log Loss": metrics["Log Loss"],
            "ROC-AUC": metrics["ROC-AUC"]
        })
    df_res = pd.DataFrame(res_records).sort_values(by="Macro F1", ascending=False)
    df_res.to_csv(out_dir / "baseline_results.csv", index=False)
    
    # Per-Class Results CSV
    pc_records = []
    for m, metrics in results.items():
        for cls_name, cls_idx in zip(["LOW", "MEDIUM", "HIGH"], [0, 1, 2]):
            pc_records.append({
                "Model": m,
                "Class": cls_name,
                "Precision": metrics["Per-Class"][cls_idx]["Precision"],
                "Recall": metrics["Per-Class"][cls_idx]["Recall"],
                "F1": metrics["Per-Class"][cls_idx]["F1"],
                "Support": metrics["Per-Class"][cls_idx]["Support"]
            })
    df_pc = pd.DataFrame(pc_records)
    df_pc.to_csv(out_dir / "per_class_results.csv", index=False)
    
    print("\n--- RESULTS ---")
    print(df_res.to_string(index=False))
    
    print("\nDone! Baseline comparison complete.")

if __name__ == "__main__":
    main()
