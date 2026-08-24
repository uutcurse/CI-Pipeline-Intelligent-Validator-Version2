import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, log_loss, brier_score_loss, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)
import xgboost as xgb
import joblib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_classical import HybridBaseline

warnings.filterwarnings('ignore')

def calculate_ece(y_true, y_prob, bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, bins + 1)
    
    for i in range(bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = in_bin | (confidences == bin_lower)
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def reliability_bins(y_true, y_prob, bin_edges):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    results = []
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        
        mask = (confidences > lower) & (confidences <= upper)
        if i == 0:
            mask = mask | (confidences == lower)
            
        n = np.sum(mask)
        acc = np.mean(accuracies[mask]) if n > 0 else np.nan
        conf = np.mean(confidences[mask]) if n > 0 else np.nan
        
        results.append({
            "Bin": f"{lower:.2f}-{upper:.2f}",
            "Count": n,
            "Accuracy": acc,
            "Avg Confidence": conf,
            "Error": conf - acc if n > 0 else np.nan
        })
    return pd.DataFrame(results)

def main():
    print("Loading data...")
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    df = df.merge(df_prov[['sample_id', 'repository']], on='sample_id', how='left')
    
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Check disjointness
    tr_repos = set(train_df['repository'])
    va_repos = set(val_df['repository'])
    te_repos = set(test_df['repository'])
    if len(tr_repos.intersection(va_repos)) > 0 or len(tr_repos.intersection(te_repos)) > 0 or len(va_repos.intersection(te_repos)) > 0:
        print("ERROR: Repositories are not disjoint.")
        sys.exit(1)
        
    text_col = "normalized_workflow_text"
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", text_col] + diagnostic_features]
    
    print("Preprocessing text and struct features...")
    # TF-IDF + SVD
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=256, random_state=42)
    
    X_tr_tfidf = tfidf.fit_transform(train_df[text_col])
    X_va_tfidf = tfidf.transform(val_df[text_col])
    X_te_tfidf = tfidf.transform(test_df[text_col])
    
    X_tr_text = svd.fit_transform(X_tr_tfidf)
    X_va_text = svd.transform(X_va_tfidf)
    X_te_text = svd.transform(X_te_tfidf)
    
    # Structural
    scaler = StandardScaler()
    X_tr_struct = scaler.fit_transform(train_df[struct_cols])
    X_va_struct = scaler.transform(val_df[struct_cols])
    X_te_struct = scaler.transform(test_df[struct_cols])
    
    # Hybrid features
    X_tr_hyb = np.hstack([X_tr_text, X_tr_struct])
    X_va_hyb = np.hstack([X_va_text, X_va_struct])
    X_te_hyb = np.hstack([X_te_text, X_te_struct])
    
    y_tr = train_df['final_label'].values
    y_va = val_df['final_label'].values
    y_te = test_df['final_label'].values
    
    candidates = {
        "C1_E06_Hybrid_LR": {
            "model": LogisticRegression(max_iter=1000, random_state=42, ),
            "X_tr": X_tr_hyb, "X_va": X_va_hyb, "X_te": X_te_hyb
        },
        "C2_N2_Hybrid_XGB": {
            "model": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                                       subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
            "X_tr": X_tr_hyb, "X_va": X_va_hyb, "X_te": X_te_hyb
        },
        "C3_Structural_RF": {
            "model": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
            "X_tr": X_tr_struct, "X_va": X_va_struct, "X_te": X_te_struct
        }
    }
    
    print("Training candidates...")
    val_records = []
    
    for name, c_dict in candidates.items():
        print(f"  Training {name}...")
        m = c_dict["model"]
        m.fit(c_dict["X_tr"], y_tr)
        
        y_va_pred = m.predict(c_dict["X_va"])
        y_va_prob = m.predict_proba(c_dict["X_va"])
        
        mac_f1 = precision_recall_fscore_support(y_va, y_va_pred, average='macro', zero_division=0)[2]
        acc = accuracy_score(y_va, y_va_pred)
        bal_acc = balanced_accuracy_score(y_va, y_va_pred)
        mcc = matthews_corrcoef(y_va, y_va_pred)
        
        val_records.append({
            "Candidate": name,
            "Val Macro F1": mac_f1,
            "Val Accuracy": acc,
            "Val Balanced Accuracy": bal_acc,
            "Val MCC": mcc
        })
        
    val_df_res = pd.DataFrame(val_records).sort_values("Val Macro F1", ascending=False)
    print("\nValidation Results:")
    print(val_df_res)
    
    # Selection rule
    best_candidate_name = val_df_res.iloc[0]['Candidate']
    print(f"\nSelected Candidate based on Validation Macro F1: {best_candidate_name}")
    
    best_c_dict = candidates[best_candidate_name]
    frozen_model = best_c_dict["model"]
    X_te_frozen = best_c_dict["X_te"]
    X_va_frozen = best_c_dict["X_va"]
    
    # 7. Calibration on Validation
    print("Fitting Calibrators on Validation...")
    # Uncalibrated
    y_va_prob_uncal = frozen_model.predict_proba(X_va_frozen)
    ll_uncal = log_loss(y_va, y_va_prob_uncal)
    
    calibrators = {
        "Uncalibrated": None,
        "Sigmoid": CalibratedClassifierCV(estimator=FrozenEstimator(frozen_model), method='sigmoid'),
        "Isotonic": CalibratedClassifierCV(estimator=FrozenEstimator(frozen_model), method='isotonic')
    }
    
    calib_val_records = []
    
    for c_name, calibrator in calibrators.items():
        if c_name == "Uncalibrated":
            probs = y_va_prob_uncal
        else:
            calibrator.fit(X_va_frozen, y_va)
            probs = calibrator.predict_proba(X_va_frozen)
            
        ll = log_loss(y_va, probs)
        ece = calculate_ece(y_va, probs)
        
        calib_val_records.append({
            "Calibration Method": c_name,
            "Val Log Loss": ll,
            "Val ECE": ece
        })
        
    calib_df_res = pd.DataFrame(calib_val_records).sort_values("Val Log Loss", ascending=True)
    best_calib_name = calib_df_res.iloc[0]["Calibration Method"]
    print(f"Selected Calibration based on Val Log Loss: {best_calib_name}")
    
    final_calibrator = calibrators[best_calib_name]
    
    # 6. Evaluate Exactly Once on Test
    print("Evaluating exactly ONCE on TEST...")
    # Original uncalibrated predictions for class logic (to avoid changing threshold boundaries)
    y_te_pred_uncal = frozen_model.predict(X_te_frozen)
    y_te_prob_uncal = frozen_model.predict_proba(X_te_frozen)
    
    if final_calibrator is not None:
        y_te_prob_calib = final_calibrator.predict_proba(X_te_frozen)
    else:
        y_te_prob_calib = y_te_prob_uncal
        
    y_te_pred_final = y_te_pred_uncal # Do NOT silently change decision threshold/classes
    
    # Metrics
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, y_te_pred_final, average=None)
    mac_pr, mac_rc, mac_f1, _ = precision_recall_fscore_support(y_te, y_te_pred_final, average='macro')
    wt_f1 = precision_recall_fscore_support(y_te, y_te_pred_final, average='weighted')[2]
    acc = accuracy_score(y_te, y_te_pred_final)
    bal_acc = balanced_accuracy_score(y_te, y_te_pred_final)
    mcc = matthews_corrcoef(y_te, y_te_pred_final)
    
    ll_uncal_te = log_loss(y_te, y_te_prob_uncal)
    ll_calib_te = log_loss(y_te, y_te_prob_calib)
    ece_calib = calculate_ece(y_te, y_te_prob_calib)
    
    # Robustness (Seeds 42, 123, 2026) for selected candidate model
    print("Running robustness seeds...")
    seeds = [42, 123, 2026]
    seed_val_f1s = []
    for s in seeds:
        if "XGB" in best_candidate_name:
            sm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                                   subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1)
        elif "LR" in best_candidate_name:
            sm = LogisticRegression(max_iter=1000, random_state=s, )
        else:
            sm = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=s, n_jobs=-1)
            
        sm.fit(best_c_dict["X_tr"], y_tr)
        sp = sm.predict(best_c_dict["X_va"])
        seed_val_f1s.append(precision_recall_fscore_support(y_va, sp, average='macro', zero_division=0)[2])
        
    mean_val_f1 = np.mean(seed_val_f1s)
    std_val_f1 = np.std(seed_val_f1s)
    
    # Error comparison against E06
    e06_model = HybridBaseline.load("experiments/e06_hybrid_logreg/model.joblib")
    y_pred_e06 = e06_model.predict(test_df[text_col], test_df[struct_cols])
    
    cor_c = (y_te_pred_final == y_te)
    cor_e = (y_pred_e06 == y_te)
    both_c = (cor_c & cor_e).sum()
    both_w = (~cor_c & ~cor_e).sum()
    c_fix_e = (cor_c & ~cor_e).sum()
    e_fix_c = (~cor_c & cor_e).sum()
    
    # High confidence errors
    conf_c = np.max(y_te_prob_calib, axis=1)
    conf_e = np.max(e06_model.model.predict_proba(e06_model.transform(test_df[text_col], test_df[struct_cols])), axis=1)
    
    hc_err_c = ((~cor_c) & (conf_c > 0.8)).sum()
    hc_err_e = ((~cor_e) & (conf_e > 0.8)).sum()
    
    # Confidence Reliability
    bins = [0.33, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    rel_uncal = reliability_bins(y_te, y_te_prob_uncal, bins)
    rel_cal = reliability_bins(y_te, y_te_prob_calib, bins)
    
    out_dir = Path("experiments/final_candidate")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    val_df_res.to_csv(out_dir / "validation_results.csv", index=False)
    calib_df_res.to_csv(out_dir / "calibration_results.csv", index=False)
    rel_uncal.to_csv(out_dir / "confidence_reliability_uncal.csv", index=False)
    rel_cal.to_csv(out_dir / "confidence_reliability.csv", index=False)
    
    test_res = {
        "Accuracy": acc,
        "Macro Precision": mac_pr,
        "Macro Recall": mac_rc,
        "Macro F1": mac_f1,
        "Weighted F1": wt_f1,
        "Balanced Accuracy": bal_acc,
        "MCC": mcc,
        "Log Loss Uncalibrated": ll_uncal_te,
        "Log Loss Calibrated": ll_calib_te,
        "ECE Calibrated": ece_calib
    }
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(test_res, f, indent=4)
        
    pc_res = []
    classes = ["LOW", "MEDIUM", "HIGH"]
    for i, c in enumerate(classes):
        pc_res.append({"Class": c, "Precision": pr[i], "Recall": rc[i], "F1": f1[i]})
    pd.DataFrame(pc_res).to_csv(out_dir / "per_class_results.csv", index=False)
    
    pd.DataFrame([
        {"Category": "Both Correct", "Count": both_c},
        {"Category": "Both Wrong", "Count": both_w},
        {"Category": f"{best_candidate_name} fixes E06", "Count": c_fix_e},
        {"Category": f"E06 fixes {best_candidate_name}", "Count": e_fix_c}
    ]).to_csv(out_dir / "error_overlap.csv", index=False)
    
    # Plots
    plt.figure(figsize=(8,6))
    sns.barplot(x="Candidate", y="Val Macro F1", data=val_df_res)
    plt.title("Candidate Selection (Validation)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / "model_selection.png")
    plt.close()
    
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_te, y_te_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix: {best_candidate_name}")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png")
    plt.close()
    
    plt.figure(figsize=(10,5))
    w = 0.35
    x = np.arange(len(rel_cal['Bin']))
    plt.bar(x - w/2, rel_uncal['Accuracy'], width=w, label='Uncalibrated Acc')
    plt.bar(x + w/2, rel_cal['Accuracy'], width=w, label='Calibrated Acc')
    plt.plot(x, rel_cal['Avg Confidence'], color='red', marker='o', label='Calibrated Conf')
    plt.xticks(x, rel_cal['Bin'])
    plt.title("Confidence Reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "confidence_reliability.png")
    plt.close()
    
    # Reports
    sel_report = f"""# Final Candidate Selection Report

### 1. Candidate Models
* C1: E06-equivalent Hybrid Logistic Regression
* C2: N2-equivalent Hybrid XGBoost
* C3: Structural Random Forest

### 2. Validation Results
* **C1 (LR) Val Macro F1:** {val_df_res[val_df_res['Candidate']=='C1_E06_Hybrid_LR']['Val Macro F1'].values[0]:.4f}
* **C2 (XGB) Val Macro F1:** {val_df_res[val_df_res['Candidate']=='C2_N2_Hybrid_XGB']['Val Macro F1'].values[0]:.4f}
* **C3 (RF) Val Macro F1:** {val_df_res[val_df_res['Candidate']=='C3_Structural_RF']['Val Macro F1'].values[0]:.4f}

### 3. Selection Rule
The candidate with the highest validation Macro F1 was selected.

### 4. Selected Candidate
**{best_candidate_name}**

### 5. Final Test Results (Evaluated ONCE)
* **Macro F1:** {mac_f1:.4f}
* **Accuracy:** {acc:.4f}

### 6. Comparison against E06
* **Selected Candidate F1:** {mac_f1:.4f}
* **E06 F1:** {precision_recall_fscore_support(y_te, y_pred_e06, average='macro')[2]:.4f}

### 7. MEDIUM-class Performance
* **Precision:** {pr[1]:.4f}
* **Recall:** {rc[1]:.4f}
* **F1:** {f1[1]:.4f}

### 8. Seed Robustness (Validation)
* **Mean Val Macro F1:** {mean_val_f1:.4f}
* **Std Val Macro F1:** {std_val_f1:.4f}

### 9. Limitations
The final candidate was selected using validation performance; the test set was used only for final evaluation.
"""
    with open(out_dir / "selection_report.md", "w") as f:
        f.write(sel_report)
        
    calib_report = f"""# Calibration Report

### Calibration Method Selection
Fitted on VALIDATION only to prevent Test set leakage.
* **Uncalibrated Val Log Loss:** {ll_uncal:.4f}
* **Sigmoid (Platt) Val Log Loss:** {calib_df_res[calib_df_res['Calibration Method']=='Sigmoid']['Val Log Loss'].values[0]:.4f}
* **Isotonic Val Log Loss:** {calib_df_res[calib_df_res['Calibration Method']=='Isotonic']['Val Log Loss'].values[0]:.4f}
* **Selected Method:** {best_calib_name}

### Test Evaluation
* **Uncalibrated Test Log Loss:** {ll_uncal_te:.4f}
* **Calibrated Test Log Loss:** {ll_calib_te:.4f}
* **Test ECE:** {ece_calib:.4f}

### Analysis
* **Did confidence reliability improve?** {'Yes' if ll_calib_te < ll_uncal_te else 'No'}. 
* **Did high-confidence errors decrease compared to E06?** Selected model has {hc_err_c} high-confidence (>0.8) errors, whereas E06 had {hc_err_e} high-confidence errors.
"""
    with open(out_dir / "calibration_report.md", "w") as f:
        f.write(calib_report)
        
    # Save Models
    joblib.dump(final_calibrator if final_calibrator else frozen_model, out_dir / "calibrated_model.joblib")
    joblib.dump(frozen_model, out_dir / "model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(tfidf, out_dir / "tfidf.joblib")
    joblib.dump(svd, out_dir / "svd.joblib")
    
    with open(out_dir / "training_config.json", "w") as f:
        json.dump({"Candidate": best_candidate_name, "Calibrator": best_calib_name}, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
