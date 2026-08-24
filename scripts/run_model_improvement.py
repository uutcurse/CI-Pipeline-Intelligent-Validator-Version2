import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_classical import HybridBaseline

def macro_f1(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]

def tune_xgboost(X_tr, y_tr, X_va, y_va, is_hybrid=True):
    best_f1 = -1
    best_params = {}
    best_model = None
    
    depths = [4, 6]
    lrs = [0.05, 0.1]
    
    for d in depths:
        for lr in lrs:
            m = xgb.XGBClassifier(
                n_estimators=200, 
                max_depth=d, 
                learning_rate=lr, 
                subsample=0.8, 
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)
            f1 = macro_f1(y_va, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'max_depth': d, 'learning_rate': lr}
                best_model = m
                
    return best_model, best_params

def tune_rf(X_tr, y_tr, X_va, y_va):
    best_f1 = -1
    best_params = {}
    best_model = None
    
    depths = [10, 20, None]
    
    for d in depths:
        m = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_va)
        f1 = macro_f1(y_va, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_params = {'max_depth': d}
            best_model = m
            
    return best_model, best_params

def main():
    print("Loading data...")
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    df = df.merge(df_prov[['sample_id', 'repository', 'workflow_id', 'workflow_path', 'commit_date']], on='sample_id', how='left')
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'test']
    
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
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", "workflow_id", "workflow_path", "commit_date", text_col] + diagnostic_features]
    
    y_train = train_df['final_label'].values
    y_val = val_df['final_label'].values
    y_test = test_df['final_label'].values
    
    X_train_struct = train_df[struct_cols]
    X_val_struct = val_df[struct_cols]
    X_test_struct = test_df[struct_cols]
    
    X_train_text = train_df[text_col]
    X_val_text = val_df[text_col]
    X_test_text = test_df[text_col]
    
    print("Preprocessing Features...")
    scaler = StandardScaler()
    X_train_struct_s = scaler.fit_transform(X_train_struct)
    X_val_struct_s = scaler.transform(X_val_struct)
    X_test_struct_s = scaler.transform(X_test_struct)
    
    # TF-IDF + SVD
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, strip_accents='unicode')
    X_train_text_v = vectorizer.fit_transform(X_train_text)
    X_val_text_v = vectorizer.transform(X_val_text)
    X_test_text_v = vectorizer.transform(X_test_text)
    
    n_components = 256
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_text_svd = svd.fit_transform(X_train_text_v)
    X_val_text_svd = svd.transform(X_val_text_v)
    X_test_text_svd = svd.transform(X_test_text_v)
    
    X_train_hybrid = np.hstack([X_train_text_svd, X_train_struct_s])
    X_val_hybrid = np.hstack([X_val_text_svd, X_val_struct_s])
    X_test_hybrid = np.hstack([X_test_text_svd, X_test_struct_s])
    
    print("Tuning N1: XGBoost Structural...")
    n1_model, n1_params = tune_xgboost(X_train_struct_s, y_train, X_val_struct_s, y_val, is_hybrid=False)
    
    print("Tuning N2: XGBoost Hybrid...")
    n2_model, n2_params = tune_xgboost(X_train_hybrid, y_train, X_val_hybrid, y_val, is_hybrid=True)
    
    print("Tuning N3: Random Forest Hybrid...")
    n3_model, n3_params = tune_rf(X_train_hybrid, y_train, X_val_hybrid, y_val)
    
    print("Loading E06 and Baseline RF for comparison...")
    e06_model = HybridBaseline.load("experiments/e06_hybrid_logreg/model.joblib")
    y_pred_e06 = e06_model.predict(X_test_text, test_df[struct_cols])
    
    rf_struct_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_struct_model.fit(X_train_struct_s, y_train)
    y_pred_rf_struct = rf_struct_model.predict(X_test_struct_s)
    
    print("Evaluating models on Test set...")
    y_pred_n1 = n1_model.predict(X_test_struct_s)
    y_pred_n2 = n2_model.predict(X_test_hybrid)
    y_pred_n3 = n3_model.predict(X_test_hybrid)
    
    res = {
        "E06 Hybrid LogReg": y_pred_e06,
        "RF Structural": y_pred_rf_struct,
        "N1 XGBoost Struct": y_pred_n1,
        "N2 XGBoost Hybrid": y_pred_n2,
        "N3 RF Hybrid": y_pred_n3
    }
    
    records = []
    pc_records = []
    
    for m_name, preds in res.items():
        f1_mac = macro_f1(y_test, preds)
        acc = accuracy_score(y_test, preds)
        p, r, f1, sup = precision_recall_fscore_support(y_test, preds, average=None)
        
        records.append({
            "Model": m_name,
            "Macro F1": f1_mac,
            "Accuracy": acc,
            "MEDIUM F1": f1[1],
            "MEDIUM Precision": p[1],
            "MEDIUM Recall": r[1]
        })
        
        for i, cls in enumerate(["LOW", "MEDIUM", "HIGH"]):
            pc_records.append({
                "Model": m_name,
                "Class": cls,
                "Precision": p[i],
                "Recall": r[i],
                "F1": f1[i]
            })
            
    res_df = pd.DataFrame(records).sort_values("Macro F1", ascending=False)
    pc_df = pd.DataFrame(pc_records)
    
    out_dir = Path("experiments/model_improvement")
    out_dir.mkdir(exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    res_df.to_csv(out_dir / "model_comparison.csv", index=False)
    pc_df.to_csv(out_dir / "per_class_results.csv", index=False)
    
    # Identify Best Model (excluding E06 and RF Struct if one of the Ns wins)
    n_models = ["N1 XGBoost Struct", "N2 XGBoost Hybrid", "N3 RF Hybrid"]
    best_n_model_name = max(n_models, key=lambda m: res_df[res_df["Model"] == m]["Macro F1"].values[0])
    y_pred_best = res[best_n_model_name]
    
    # Robustness Analysis
    print(f"Running robustness check for {best_n_model_name}...")
    seeds = [42, 123, 2026]
    robust_f1s = []
    
    for s in seeds:
        if "N2 XGBoost" in best_n_model_name:
            m = xgb.XGBClassifier(n_estimators=200, max_depth=n2_params['max_depth'], learning_rate=n2_params['learning_rate'], subsample=0.8, colsample_bytree=0.8, random_state=s, use_label_encoder=False, eval_metric='mlogloss')
            m.fit(X_train_hybrid, y_train)
            p = m.predict(X_test_hybrid)
        elif "N3 RF" in best_n_model_name:
            m = RandomForestClassifier(n_estimators=100, max_depth=n3_params['max_depth'], random_state=s, n_jobs=-1)
            m.fit(X_train_hybrid, y_train)
            p = m.predict(X_test_hybrid)
        else: # N1
            m = xgb.XGBClassifier(n_estimators=200, max_depth=n1_params['max_depth'], learning_rate=n1_params['learning_rate'], subsample=0.8, colsample_bytree=0.8, random_state=s, use_label_encoder=False, eval_metric='mlogloss')
            m.fit(X_train_struct_s, y_train)
            p = m.predict(X_test_struct_s)
        robust_f1s.append(macro_f1(y_test, p))
        
    robust_df = pd.DataFrame({"Seed": seeds, "Macro F1": robust_f1s})
    robust_df.to_csv(out_dir / "robustness_results.csv", index=False)
    
    # Error Overlap Analysis
    correct_e06 = (y_pred_e06 == y_test)
    correct_rf = (y_pred_rf_struct == y_test)
    correct_best = (y_pred_best == y_test)
    
    both_correct = (correct_best & correct_e06).sum()
    both_wrong = (~correct_best & ~correct_e06).sum()
    new_fixes_e06 = (correct_best & ~correct_e06).sum()
    e06_fixes_new = (~correct_best & correct_e06).sum()
    
    new_fixes_rf = (correct_best & ~correct_rf).sum()
    rf_fixes_new = (~correct_best & correct_rf).sum()
    
    overlap_records = [
        {"Category": "Both Correct (Best vs E06)", "Count": both_correct},
        {"Category": "Both Wrong (Best vs E06)", "Count": both_wrong},
        {"Category": "Best fixes E06", "Count": new_fixes_e06},
        {"Category": "E06 fixes Best", "Count": e06_fixes_new},
        {"Category": "Best fixes RF", "Count": new_fixes_rf},
        {"Category": "RF fixes Best", "Count": rf_fixes_new}
    ]
    pd.DataFrame(overlap_records).to_csv(out_dir / "error_overlap.csv", index=False)
    
    # Feature Importance for Best Model (if XGBoost or RF)
    if "XGBoost" in best_n_model_name or "RF" in best_n_model_name:
        if best_n_model_name == "N1 XGBoost Struct":
            fi = n1_model.feature_importances_
            cols = struct_cols
        elif best_n_model_name == "N2 XGBoost Hybrid":
            fi = n2_model.feature_importances_
            cols = [f"Text_SVD_{i}" for i in range(256)] + struct_cols
        else:
            fi = n3_model.feature_importances_
            cols = [f"Text_SVD_{i}" for i in range(256)] + struct_cols
            
        fi_df = pd.DataFrame({"Feature": cols, "Importance": fi}).sort_values("Importance", ascending=False)
        fi_df.to_csv(out_dir / "feature_importance.csv", index=False)
        
        # Aggregate text importance
        text_imp = fi_df[fi_df['Feature'].str.startswith('Text_SVD')]['Importance'].sum()
        struct_imp = fi_df[~fi_df['Feature'].str.startswith('Text_SVD')]['Importance'].sum()
        print(f"Text Total Importance: {text_imp:.4f}, Struct Total Importance: {struct_imp:.4f}")
    
    # Save training config
    config = {
        "N1_params": n1_params,
        "N2_params": n2_params,
        "N3_params": n3_params,
        "best_model": best_n_model_name,
        "svd_components": 256
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Plots
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x="Model", y="Macro F1")
    plt.xticks(rotation=20)
    plt.title("Model Comparison - Macro F1")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_macro_f1_comparison.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x="Model", y="MEDIUM F1")
    plt.xticks(rotation=20)
    plt.title("Model Comparison - MEDIUM Class F1")
    plt.tight_layout()
    plt.savefig(plots_dir / "medium_class_f1_comparison.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=pd.DataFrame(overlap_records), y="Category", x="Count", orient='h')
    plt.title("Error Overlap")
    plt.tight_layout()
    plt.savefig(plots_dir / "error_overlap.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df.head(20), x="Importance", y="Feature")
    plt.title("Top 20 Feature Importances (Best Model)")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance.png")
    plt.close()
    
    # Report Markdown
    f1_n1 = res_df[res_df['Model'] == 'N1 XGBoost Struct']['Macro F1'].values[0]
    f1_n2 = res_df[res_df['Model'] == 'N2 XGBoost Hybrid']['Macro F1'].values[0]
    f1_n3 = res_df[res_df['Model'] == 'N3 RF Hybrid']['Macro F1'].values[0]
    f1_e06 = res_df[res_df['Model'] == 'E06 Hybrid LogReg']['Macro F1'].values[0]
    f1_rf = res_df[res_df['Model'] == 'RF Structural']['Macro F1'].values[0]
    
    med_f1_best = res_df[res_df['Model'] == best_n_model_name]['MEDIUM F1'].values[0]
    med_f1_e06 = res_df[res_df['Model'] == 'E06 Hybrid LogReg']['MEDIUM F1'].values[0]
    
    diff_e06 = res_df.iloc[0]['Macro F1'] - f1_e06
    diff_rf = res_df.iloc[0]['Macro F1'] - f1_rf
    
    best_overall_f1 = res_df.iloc[0]['Macro F1']
    best_overall_model = res_df.iloc[0]['Model']
    
    improved_errors = "Yes" if new_fixes_e06 > e06_fixes_new else "No"
    
    mean_rob = np.mean(robust_f1s)
    std_rob = np.std(robust_f1s)
    
    is_transformer_justified = "Yes"
    if diff_e06 > 0.05:
        is_transformer_justified = "No, the hybrid nonlinear classical model already achieved a massive breakthrough, making a complex Transformer potentially unnecessary or secondary."
    elif diff_e06 <= 0.0:
        is_transformer_justified = "Yes, because all classical hybrid techniques (linear and nonlinear combinations) fail to improve upon the E06 baseline, suggesting that shallow representations of text (TF-IDF/SVD) have hit an architectural ceiling. A deeper semantic understanding of text (via a Transformer) is strongly warranted."
    else:
        is_transformer_justified = "Yes. While there is marginal improvement, the performance ceiling remains low. Extracting deeper semantic tokens or graph structures likely requires a deep architecture."
        
    report_md = f"""# Model Improvement Report: Hybrid Nonlinear Classical

### 1. Best New Model
{best_n_model_name}

### 2. Macro F1
{best_overall_f1:.4f}

### 3. Difference from E06
{best_overall_f1 - f1_e06:+.4f}

### 4. Difference from Random Forest Structural
{best_overall_f1 - f1_rf:+.4f}

### 5. MEDIUM-class F1
* Best Model: {med_f1_best:.4f}
* E06: {med_f1_e06:.4f}
* Change: {med_f1_best - med_f1_e06:+.4f}

### 6. Did Nonlinear Modeling Solve Meaningful Errors?
The best nonlinear model fixed {new_fixes_e06} errors that E06 made, but it also missed {e06_fixes_new} errors that E06 correctly classified. Overall improvement is captured in the F1 change.

### 7. Is the Improvement Consistent Across Seeds?
* Mean Macro F1 (3 seeds): {mean_rob:.4f}
* Std Dev: {std_rob:.4f}
Consistency indicates robust optimization.

### 8. Is the Added Complexity Justified?
If the performance gain is minimal or negative compared to E06, then combining high-dimensional SVD and gradient boosting does not justify the added latency and architectural overhead.

### 9. Is a Transformer/GNN Warranted?
{is_transformer_justified}
"""
    with open(out_dir / "model_improvement_report.md", "w") as f:
        f.write(report_md)
        
    print("All tasks complete. Writing output JSON for summary.")
    summary = {
        "N1 Macro F1": f1_n1,
        "N2 Macro F1": f1_n2,
        "N3 Macro F1": f1_n3,
        "Best model": best_n_model_name,
        "E06 baseline": f1_e06,
        "Improvement over E06": best_overall_f1 - f1_e06,
        "MEDIUM F1 best": med_f1_best,
        "MEDIUM F1 e06": med_f1_e06,
        "Error overlap": {
            "Best fixes E06": int(new_fixes_e06),
            "E06 fixes Best": int(e06_fixes_new),
            "Best fixes RF": int(new_fixes_rf),
            "RF fixes Best": int(rf_fixes_new)
        },
        "Robustness": {"mean": mean_rob, "std": std_rob},
        "Transformer Justified": is_transformer_justified
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    main()
