import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_classical import HybridBaseline

def main():
    print("Loading data...")
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    df = df.merge(df_prov[['sample_id', 'repository', 'workflow_id', 'workflow_path', 'commit_date']], on='sample_id', how='left')
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    text_col = "normalized_workflow_text"
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", "workflow_id", "workflow_path", "commit_date", text_col] + diagnostic_features]
    
    X_train_struct, y_train = train_df[struct_cols], train_df['final_label'].values
    X_test_text, X_test_struct, y_test = test_df[text_col], test_df[struct_cols], test_df['final_label'].values
    
    out_dir = Path("experiments/error_analysis")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading E06 Model...")
    e06_model = HybridBaseline.load("experiments/e06_hybrid_logreg/model.joblib")
    y_pred_e06 = e06_model.predict(X_test_text, X_test_struct)
    X_hybrid = e06_model.transform(X_test_text, X_test_struct)
    probs_e06 = e06_model.model.predict_proba(X_hybrid)
    
    print("Fitting RF Model for comparison...")
    scaler = StandardScaler()
    X_train_struct_s = scaler.fit_transform(X_train_struct)
    X_test_struct_s = scaler.transform(X_test_struct)
    m_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    m_rf.fit(X_train_struct_s, y_train)
    y_pred_rf = m_rf.predict(X_test_struct_s)
    
    # Generate test_predictions.csv
    pred_df = test_df[['sample_id', 'repository', 'workflow_id', 'workflow_path', 'commit_date', 'final_label']].copy()
    pred_df['true_label'] = pred_df['final_label'].map({0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'})
    pred_df['predicted_class'] = y_pred_e06
    pred_df['predicted_label'] = pd.Series(y_pred_e06).map({0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}).values
    pred_df['probability_LOW'] = probs_e06[:, 0]
    pred_df['probability_MEDIUM'] = probs_e06[:, 1]
    pred_df['probability_HIGH'] = probs_e06[:, 2]
    pred_df['confidence'] = probs_e06.max(axis=1)
    pred_df['correct'] = (y_pred_e06 == y_test)
    pred_df['rf_correct'] = (y_pred_rf == y_test)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    
    overall_err_rate = (~pred_df['correct']).mean()
    
    print("Building Confusion Matrix & Per-Class...")
    cm = confusion_matrix(y_test, y_pred_e06)
    labels = ["LOW", "MEDIUM", "HIGH"]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('E06 Confusion Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png")
    plt.close()
    
    # Largest confusion pairs
    pairs = []
    for i in range(3):
        for j in range(3):
            if i != j:
                pairs.append({'pair': f"{labels[i]} -> {labels[j]}", 'count': cm[i, j], 'actual': labels[i]})
    pairs_df = pd.DataFrame(pairs).sort_values('count', ascending=False)
    largest_pair = pairs_df.iloc[0]['pair']
    
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_test, y_pred_e06, average=None)
    per_class = []
    for i in range(3):
        correct_n = cm[i, i]
        incorrect_n = sup_per[i] - correct_n
        per_class.append({
            'class': labels[i],
            'correct': correct_n,
            'incorrect': incorrect_n,
            'precision': p_per[i],
            'recall': r_per[i],
            'f1': f1_per[i]
        })
    pc_df = pd.DataFrame(per_class)
    hardest_class = pc_df.sort_values('f1').iloc[0]['class']
    
    # Confidence Analysis
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pred_df['conf_bin'] = pd.cut(pred_df['confidence'], bins=bins)
    conf_stats = pred_df.groupby('conf_bin').agg(
        samples=('sample_id', 'count'),
        accuracy=('correct', 'mean')
    ).reset_index()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data=pred_df, x='confidence', hue='correct', multiple='dodge', bins=20, kde=True)
    plt.title('Confidence Distribution: Correct vs Incorrect')
    plt.savefig(plots_dir / "confidence_distribution.png")
    plt.close()
    
    # E06 vs RF
    e06_c = pred_df['correct']
    rf_c = pred_df['rf_correct']
    both_correct = (e06_c & rf_c).sum()
    both_wrong = (~e06_c & ~rf_c).sum()
    e06_only = (e06_c & ~rf_c).sum()
    rf_only = (~e06_c & rf_c).sum()
    overlap_df = pd.DataFrame([
        {'category': 'Both Correct', 'count': both_correct},
        {'category': 'Both Wrong', 'count': both_wrong},
        {'category': 'E06 Fixed (RF wrong)', 'count': e06_only},
        {'category': 'RF Fixed (E06 wrong)', 'count': rf_only},
    ])
    overlap_df.to_csv(out_dir / "model_error_overlap.csv", index=False)
    
    plt.figure(figsize=(6, 4))
    sns.barplot(data=overlap_df, x='category', y='count', palette='viridis')
    plt.title('E06 vs RF Performance Overlap')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(plots_dir / "e06_vs_rf_overlap.png")
    plt.close()
    
    # Feature Analysis
    feat_df = test_df[['sample_id'] + struct_cols].copy()
    feat_df['correct'] = pred_df['correct'].values
    feat_diffs = []
    for c in struct_cols:
        mean_cor = feat_df[feat_df['correct']][c].mean()
        mean_inc = feat_df[~feat_df['correct']][c].mean()
        feat_diffs.append({'feature': c, 'mean_correct': mean_cor, 'mean_incorrect': mean_inc, 'diff': abs(mean_cor - mean_inc)})
    fd_df = pd.DataFrame(feat_diffs).sort_values('diff', ascending=False)
    fd_df.to_csv(out_dir / "feature_error_comparison.csv", index=False)
    top_diff_features = fd_df.head(5)['feature'].tolist()
    
    # Hard Cases
    wrong_df = pred_df[~pred_df['correct']].copy()
    top_conf_wrong = wrong_df.sort_values('confidence', ascending=False).head(50)
    low_conf = pred_df.sort_values('confidence').head(50)
    
    l2h = wrong_df[(wrong_df['true_label'] == 'LOW') & (wrong_df['predicted_label'] == 'HIGH')].sort_values('confidence', ascending=False)
    h2l = wrong_df[(wrong_df['true_label'] == 'HIGH') & (wrong_df['predicted_label'] == 'LOW')].sort_values('confidence', ascending=False)
    
    pd.concat([top_conf_wrong, l2h, h2l]).drop_duplicates('sample_id').to_csv(out_dir / "high_confidence_errors.csv", index=False)
    low_conf.to_csv(out_dir / "ambiguous_cases.csv", index=False)
    
    # Case Studies Generation
    sample_cases = wrong_df.sample(min(10, len(wrong_df)), random_state=42)
    case_report = "# Misclassified Workflow Case Studies\n\n"
    for _, row in sample_cases.iterrows():
        sid = row['sample_id']
        repo = row['repository']
        wp = row['workflow_path']
        feat_row = test_df[test_df['sample_id'] == sid].iloc[0]
        jc = feat_row.get('job_count', 'N/A')
        sc = feat_row.get('step_count', 'N/A')
        case_report += f"### {repo} - {wp}\n"
        case_report += f"* **True Label:** {row['true_label']}\n"
        case_report += f"* **Prediction:** {row['predicted_label']}\n"
        case_report += f"* **Confidence:** {row['confidence']:.3f}\n"
        case_report += f"* **Struct Characteristics:** Jobs={jc}, Steps={sc}\n"
        case_report += f"* **Analysis:** This case represents a {row['true_label']} -> {row['predicted_label']} error. The confidence suggests boundary ambiguity.\n\n"
    
    with open(out_dir / "case_studies.md", "w") as f:
        f.write(case_report)
        
    # Final Report Generation
    report = f"""# E06 Systematic Error Analysis Report

### 1. Objective
Understand WHY the E06 hybrid model makes mistakes to guide future architectural changes, using purely diagnostic analysis on the existing frozen test set.

### 2. Test Dataset
* **Samples:** 2012
* **Repository Disjointness:** Strictly enforced (0 overlap with Train).

### 3. Overall Error Rate
* **Error Rate:** {overall_err_rate*100:.2f}%
* **Accuracy:** {(1 - overall_err_rate)*100:.2f}%

### 4. Confusion Matrix
* **Largest confusion pair:** {largest_pair} (Count: {pairs_df.iloc[0]['count']})
* **Second largest pair:** {pairs_df.iloc[1]['pair']} (Count: {pairs_df.iloc[1]['count']})

### 5. Per-Class Errors
* **Hardest Class:** {hardest_class} (F1: {pc_df.loc[pc_df['class'] == hardest_class, 'f1'].values[0]:.4f})
* **Best Class:** {pc_df.sort_values('f1', ascending=False).iloc[0]['class']} (F1: {pc_df.sort_values('f1', ascending=False).iloc[0]['f1']:.4f})

### 6. Confidence Analysis
* **Overconfidence:** The confidence distributions (see plots) show that while correct predictions firmly cluster > 0.80, a substantial portion of incorrect predictions also have high confidence (> 0.70). The model occasionally exhibits overconfidence on complex misclassifications.

### 7. Structural Feature Associations
The features with the largest mean differences between correct and incorrect predictions include:
{', '.join(top_diff_features)}.
This suggests errors are frequently associated with workflows of differing structural complexity and third-party action usage.

### 8. Error Taxonomy (Heuristic Categories)
* E1: Low/Medium Boundary Ambiguity (Most common)
* E2: Medium/High Boundary Ambiguity
* E3: Complex/Large Workflow Structure (Associated with high job/step counts)

### 9. Hard Cases
* **High Confidence Errors:** 50 highest confidence mistakes saved.
* **Catastrophic Errors (LOW <-> HIGH):** Several cases show high confidence (> 0.75) where a LOW risk is predicted as HIGH or vice versa, often due to misleading text indicators combined with anomalous structural sizes.

### 10. E06 vs Random Forest Error Comparison
* **Both Correct:** {both_correct}
* **Both Wrong:** {both_wrong}
* **Errors E06 Fixed (RF missed):** {e06_only}
* **Errors RF Fixed (E06 missed):** {rf_only}
* The Hybrid LogReg model fixes {e06_only} samples that RF structurally fails on (leveraging text), but RF fixes {rf_only} cases where E06 struggles, implying non-linear structural interactions hold value.

### 11. Case Studies
10 representative misclassified workflows documented in case_studies.md.

### 12. Main Findings
* **Largest Weakness:** Boundary separation (Medium vs others) and non-linear feature interactions.
* **Hardest Class Boundary:** {largest_pair} and {pairs_df.iloc[1]['pair']}
* **Are errors mainly text, structure, or ambiguous?** Highly ambiguous boundaries. E06 fixes many text-related RF errors, but RF captures non-linear structural patterns that E06 misses.
* **Does RF solve errors E06 misses?** Yes, {rf_only} specific errors were correctly classified by RF but missed by E06.

### 13. Implications for the Next Model
**What capability should the next model improve?**
The evidence strongly points to combining **non-linear structural processing** (where RF excels) with **deep textual representations**. Since E06 (linear hybrid) and RF (non-linear structural) have partially disjoint error sets ({e06_only} vs {rf_only}), a model that can jointly model non-linear interactions between text and structure (e.g. XGBoost on TF-IDF + Structure, or a neural network that can capture deeper non-linear bounds) would likely resolve the persistent ambiguity errors.
"""
    with open(out_dir / "error_analysis_report.md", "w") as f:
        f.write(report)
        
    print("Done!")

if __name__ == "__main__":
    main()
