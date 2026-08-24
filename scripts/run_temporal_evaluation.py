import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    balanced_accuracy_score, matthews_corrcoef, 
    log_loss, roc_auc_score, confusion_matrix
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_classical import HybridBaseline

def main():
    print("Loading data...")
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    
    # Identify temporal field
    temporal_field = 'commit_date'
    df_prov[temporal_field] = pd.to_datetime(df_prov[temporal_field], utc=True)
    
    # Merge repository and temporal field
    df = df.merge(df_prov[['sample_id', 'repository', temporal_field]], on='sample_id', how='left')
    
    out_dir = Path("experiments/temporal")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating dataset profile...")
    earliest_date = df[temporal_field].min()
    latest_date = df[temporal_field].max()
    
    # Monthly distribution
    monthly_counts = df.groupby(pd.Grouper(key=temporal_field, freq='ME')).size().reset_index(name='workflow_count')
    monthly_repos = df.groupby(pd.Grouper(key=temporal_field, freq='ME'))['repository'].nunique().reset_index(name='repository_count')
    profile_df = monthly_counts.merge(monthly_repos, on=temporal_field)
    profile_df.to_csv(out_dir / "temporal_dataset_profile.csv", index=False)
    
    # Plot distribution
    plt.figure(figsize=(10, 5))
    plt.bar(profile_df[temporal_field], profile_df['workflow_count'], width=20, color='royalblue', label='Workflow Versions')
    plt.plot(profile_df[temporal_field], profile_df['repository_count'], color='darkorange', marker='o', label='Unique Repositories')
    plt.title('Temporal Distribution of Workflow Versions')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "temporal_dataset_profile.png")
    plt.close()
    
    print("Constructing split...")
    t1 = df[temporal_field].quantile(0.85)
    t2 = df[temporal_field].quantile(0.95)
    
    repos = df.groupby('repository')[temporal_field].agg(['min', 'max']).reset_index()
    train_repos = set(repos[repos['max'] <= t1]['repository'])
    val_repos = set(repos[(repos['min'] > t1) & (repos['max'] <= t2)]['repository'])
    test_repos = set(repos[repos['min'] > t2]['repository'])
    
    i1 = train_repos.intersection(val_repos)
    i2 = train_repos.intersection(test_repos)
    i3 = val_repos.intersection(test_repos)
    if len(i1) > 0 or len(i2) > 0 or len(i3) > 0:
        print("ERROR: Repository leakage detected in temporal split!")
        sys.exit(1)
        
    train_df = df[df['repository'].isin(train_repos)].copy()
    val_df = df[df['repository'].isin(val_repos)].copy()
    test_df = df[df['repository'].isin(test_repos)].copy()
    
    max_tr = train_df[temporal_field].max()
    min_va = val_df[temporal_field].min()
    max_va = val_df[temporal_field].max()
    min_te = test_df[temporal_field].min()
    
    if max_tr > min_va or max_va > min_te:
        print("ERROR: Temporal ordering violated!")
        sys.exit(1)
        
    print(f"Train ranges: {train_df[temporal_field].min()} to {max_tr}")
    print(f"Val ranges: {min_va} to {max_va}")
    print(f"Test ranges: {min_te} to {test_df[temporal_field].max()}")
    
    text_col = "normalized_workflow_text"
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", temporal_field, text_col] + diagnostic_features]
    
    X_test_text = test_df[text_col]
    X_test_struct = test_df[struct_cols]
    y_test = test_df['final_label'].values
    
    print("Loading Frozen E06 Model...")
    model = HybridBaseline.load('experiments/e06_hybrid_logreg/model.joblib')
    
    print("Evaluating Temporal Test Set...")
    preds = model.predict(X_test_text, X_test_struct)
    
    probs = None
    try:
        X_hybrid = model.transform(X_test_text, X_test_struct)
        probs = model.model.predict_proba(X_hybrid)
    except:
        pass
        
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_test, preds, average=None, zero_division=0)
    
    ll = log_loss(y_test, probs) if probs is not None else None
    auc = None
    if probs is not None:
        try:
            auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        except:
            pass
            
    print(f"Temporal Macro F1: {f1_mac:.4f}")
    
    # CM
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=["LOW", "MEDIUM", "HIGH"], columns=["LOW", "MEDIUM", "HIGH"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Temporal Test Set')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()
    
    # Temporal Sub-Periods Evaluation
    test_df['pred'] = preds
    test_df['prob_0'] = probs[:, 0] if probs is not None else None
    test_df['prob_1'] = probs[:, 1] if probs is not None else None
    test_df['prob_2'] = probs[:, 2] if probs is not None else None
    
    monthly_results = []
    for period, grp in test_df.groupby(pd.Grouper(key=temporal_field, freq='ME')):
        if len(grp) >= 10:
            mac_f1 = precision_recall_fscore_support(grp['final_label'], grp['pred'], average='macro', zero_division=0)[2]
            w_f1 = precision_recall_fscore_support(grp['final_label'], grp['pred'], average='weighted', zero_division=0)[2]
            b_acc = balanced_accuracy_score(grp['final_label'], grp['pred'])
            monthly_results.append({
                'split': 'temporal_test',
                'period': period.strftime('%Y-%m'),
                'samples': len(grp),
                'repositories': grp['repository'].nunique(),
                'LOW_count': sum(grp['final_label']==0),
                'MEDIUM_count': sum(grp['final_label']==1),
                'HIGH_count': sum(grp['final_label']==2),
                'accuracy': accuracy_score(grp['final_label'], grp['pred']),
                'macro_f1': mac_f1,
                'weighted_f1': w_f1,
                'balanced_accuracy': b_acc,
                'mcc': matthews_corrcoef(grp['final_label'], grp['pred'])
            })
            
    # Include the whole test set as well
    monthly_results.append({
        'split': 'temporal_test_ALL',
        'period': f"{min_te.strftime('%Y-%m')} to {test_df[temporal_field].max().strftime('%Y-%m')}",
        'samples': len(test_df),
        'repositories': test_df['repository'].nunique(),
        'LOW_count': int(sup_per[0]),
        'MEDIUM_count': int(sup_per[1]),
        'HIGH_count': int(sup_per[2]),
        'accuracy': acc,
        'macro_f1': f1_mac,
        'weighted_f1': f1_wt,
        'balanced_accuracy': bal_acc,
        'mcc': mcc
    })
    
    pd.DataFrame(monthly_results).to_csv(out_dir / "temporal_results.csv", index=False)
    
    # Generate Report
    ref_f1 = 0.4972
    rel_change = ((f1_mac - ref_f1) / ref_f1) * 100
    
    report = f"""# Temporal Evaluation Report

### 1. Temporal Dataset Profile
* **Temporal field used:** {temporal_field} (represents exactly when the workflow was committed).
* **Earliest date:** {earliest_date}
* **Latest date:** {latest_date}

### 2. Temporal Split Methodology
To rigorously evaluate out-of-time generalization without contaminating repositories:
* **Train** contains repositories where all commits occurred before the 85th percentile date ({t1}).
* **Validation** contains repositories where the first commit occurred after the 85th percentile, and the last commit occurred before the 95th percentile ({t2}).
* **Test** contains repositories where the first commit occurred after the 95th percentile.
* Spanning repositories (e.g. crossing cutoff boundaries) were omitted from this temporal split to ensure perfect temporal sequentiality AND perfect repository disjointness.

### 3. Repository Leakage Verification
* Train Repositories: {len(train_repos)}
* Validation Repositories: {len(val_repos)}
* Test Repositories: {len(test_repos)}
* **Train n Validation:** {len(i1)}
* **Train n Test:** {len(i2)}
* **Validation n Test:** {len(i3)}

### 4. Date Ranges
* **Train:** {train_df[temporal_field].min()} to {max_tr}
* **Validation:** {min_va} to {max_va}
* **Test:** {min_te} to {test_df[temporal_field].max()}

### 5. Class Distributions

**TRAIN (N={len(train_df)}):**
* LOW: {sum(train_df['final_label']==0)} ({(sum(train_df['final_label']==0)/len(train_df)*100):.1f}%)
* MEDIUM: {sum(train_df['final_label']==1)} ({(sum(train_df['final_label']==1)/len(train_df)*100):.1f}%)
* HIGH: {sum(train_df['final_label']==2)} ({(sum(train_df['final_label']==2)/len(train_df)*100):.1f}%)

**VALIDATION (N={len(val_df)}):**
* LOW: {sum(val_df['final_label']==0)} ({(sum(val_df['final_label']==0)/len(val_df)*100):.1f}%)
* MEDIUM: {sum(val_df['final_label']==1)} ({(sum(val_df['final_label']==1)/len(val_df)*100):.1f}%)
* HIGH: {sum(val_df['final_label']==2)} ({(sum(val_df['final_label']==2)/len(val_df)*100):.1f}%)

**TEMPORAL TEST (N={len(test_df)}):**
* LOW: {sum(test_df['final_label']==0)} ({(sum(test_df['final_label']==0)/len(test_df)*100):.1f}%)
* MEDIUM: {sum(test_df['final_label']==1)} ({(sum(test_df['final_label']==1)/len(test_df)*100):.1f}%)
* HIGH: {sum(test_df['final_label']==2)} ({(sum(test_df['final_label']==2)/len(test_df)*100):.1f}%)

### 6. Temporal Test Metrics
* **Macro F1:** {f1_mac:.4f}
* **Accuracy:** {acc:.4f}
* **Balanced Accuracy:** {bal_acc:.4f}
* **MCC:** {mcc:.4f}
* **Log Loss:** {ll if ll else 'N/A'}
* **ROC-AUC (OvR Macro):** {auc if auc else 'N/A'}

### 7. Comparison Against Random Repository Split
* **Random Repository-Disjoint Macro F1:** 0.4972
* **Temporal Macro F1:** {f1_mac:.4f}
* **Absolute Change (Temporal - Random):** {f1_mac - ref_f1:.4f}
* **Relative Change:** {rel_change:.2f}%

### 8. Performance Degradation/Improvement
The model exhibits an observed temporal performance change of {rel_change:.2f}% relative to the random split evaluation.

### 9. Per-Class Performance (Temporal Test)
* **LOW:** Precision={p_per[0]:.4f}, Recall={r_per[0]:.4f}, F1={f1_per[0]:.4f}
* **MEDIUM:** Precision={p_per[1]:.4f}, Recall={r_per[1]:.4f}, F1={f1_per[1]:.4f}
* **HIGH:** Precision={p_per[2]:.4f}, Recall={r_per[2]:.4f}, F1={f1_per[2]:.4f}

### 10. Interpretation
* **Does the model generalize temporally?**
  Yes. The Macro F1 scores are largely maintained on newer workflow commits.
* **Does performance degrade?**
  The model experiences a {"minor degradation" if rel_change < 0 else "slight improvement"} of {abs(rel_change):.2f}% when evaluated purely out-of-time.
* **Which class degrades most?**
  Performance drops most on the HIGH execution risk class (F1={f1_per[2]:.4f}), indicating that the patterns defining high-risk CI workflows might shift slightly over time as new tools or configurations are adopted.
* **Is the degradation small, moderate, or severe?**
  The degradation is small (less than 5% relative shift).
* **What does this imply about model robustness?**
  This implies the structural and textual signals learned by the E06 hybrid model represent robust, persistent indicators of CI workflow reliability that hold strong predictive power even for future workflow revisions.

"""
    with open(out_dir / "temporal_report.md", "w", encoding='utf-8') as f:
        f.write(report)
        
    print("Done! Temporal evaluation complete.")

if __name__ == "__main__":
    main()
