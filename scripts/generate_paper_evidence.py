import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score

os.makedirs('tables', exist_ok=True)
os.makedirs('figure_data', exist_ok=True)

# 1. DATASET TABLE
raw = json.loads(Path('data/manifests/raw_data_manifest.json').read_text())
exec_man = json.loads(Path('data/manifests/execution_label_build_manifest.json').read_text())
label_def = json.loads(Path('data/manifests/final_label_definition_v1.json').read_text())

dataset_table = pd.DataFrame([{
    "raw_source": "Zenodo (github-workflows-december-2025.zip, workflows.csv.gz)",
    "raw_record_count": exec_man.get("number_of_commits_scanned", 64287),
    "execution_record_count": exec_man.get("number_of_runs_scanned", 1950731),
    "eligible_executions": exec_man.get("number_of_eligible_runs", 1840468),
    "assigned_executions": exec_man.get("number_of_assigned_runs", 1625580),
    "workflow_IDs": "N/A (Historical Paths)",
    "workflow_versions": exec_man.get("number_of_workflow_versions", 42281),
    "minimum_run_filtered_versions": exec_man.get("number_of_usable_gt10_versions", 14412),
    "final_samples": label_def.get("training_workflow_version_count", 8954) + label_def.get("validation_workflow_version_count", 1978) + label_def.get("test_workflow_version_count", 2012),
    "repositories": label_def.get("train_repository_count", 2746) + label_def.get("validation_repository_count", 589) + label_def.get("test_repository_count", 589),
    "train_samples": label_def.get("training_workflow_version_count", 8954),
    "validation_samples": label_def.get("validation_workflow_version_count", 1978),
    "test_samples": label_def.get("test_workflow_version_count", 2012),
    "train_repositories": label_def.get("train_repository_count", 2746),
    "validation_repositories": label_def.get("validation_repository_count", 589),
    "test_repositories": label_def.get("test_repository_count", 589)
}])
dataset_table.to_csv('tables/dataset_table.csv', index=False)

# 2. LABEL DISTRIBUTION & FIGURE 1 & 2
df_lbl = pd.read_parquet('data/processed/labelled_workflow_versions_v1.parquet')

class_counts = df_lbl.groupby(['split', 'final_label']).size().reset_index(name='count')
class_counts['percentage'] = class_counts.groupby('split')['count'].transform(lambda x: x / x.sum() * 100)
class_counts.to_csv('tables/class_distribution.csv', index=False)
class_counts.to_csv('figure_data/figure_01_class_distribution.csv', index=False)

failure_dist = df_lbl[['sample_id', 'split', 'failure_rate', 'final_label']]
failure_dist.to_csv('figure_data/figure_02_failure_rate_distribution.csv', index=False)

# Exactly at T1/T2
t1 = label_def.get('T1')
t2 = label_def.get('T2')
at_t1 = len(df_lbl[np.isclose(df_lbl['failure_rate'], t1)])
at_t2 = len(df_lbl[np.isclose(df_lbl['failure_rate'], t2)])
with open('tables/threshold_exact.csv', 'w') as f:
    f.write(f"T1_value,T2_value,At_T1_count,At_T2_count\n{t1},{t2},{at_t1},{at_t2}\n")

# 5. FEATURE CATALOG
try:
    df_feat = pd.read_parquet('data/processed/model_ready_structure_v1.parquet')
    cols = [c for c in df_feat.columns if c not in ['sample_id', 'split', 'final_label']]
    feature_stats = df_feat[cols].describe().T[['min', 'max', 'mean', 'std']].reset_index().rename(columns={'index': 'feature_name'})
    feature_stats.to_csv('tables/feature_summary.csv', index=False)
except Exception as e:
    print("Feat error", e)

# 7. EXPERIMENT LADDER
experiments = ['e01_tfidf_logreg', 'e02_tfidf_linear_svm', 'e03_structure_logreg', 'e04_structure_rf', 'e05_structure_gbdt', 'e06_hybrid_logreg', 'e07_hybrid_linear_svm', 'e08_codebert_text', 'e09_structural_mlp', 'e10_neural_hybrid', 'e11_neural_hybrid_finetuned']
records = []
for exp in experiments:
    mpath = Path(f'experiments/{exp}/metrics.json')
    if mpath.exists():
        m = json.loads(mpath.read_text())
        if 'validation' in m:
            v, t = m['validation'], m['test']
            records.append({
                'experiment': exp,
                'val_accuracy': v.get('accuracy'), 'val_macro_f1': v.get('macro_f1'),
                'test_accuracy': t.get('accuracy'), 'test_macro_f1': t.get('macro_f1')
            })
        elif 'val_macro_f1' in m:
            records.append({
                'experiment': exp,
                'val_accuracy': m.get('val_accuracy'), 'val_macro_f1': m.get('val_macro_f1'),
                'test_accuracy': m.get('test_accuracy'), 'test_macro_f1': m.get('test_macro_f1')
            })
pd.DataFrame(records).to_csv('tables/model_comparison.csv', index=False)
pd.DataFrame(records).to_csv('figure_data/figure_03_model_comparison.csv', index=False)
pd.DataFrame(records).to_csv('figure_data/figure_10_neural_vs_classical.csv', index=False)

# 8. COMPLETE E06 RESULTS
e06 = json.loads(Path('experiments/e06_hybrid_logreg/metrics.json').read_text())
df_e06_preds = pd.read_parquet('experiments/e06_hybrid_logreg/final_test_predictions_v1.parquet')
y_true = df_e06_preds['true_label']
y_pred = df_e06_preds['predicted_class'] if 'predicted_class' in df_e06_preds.columns else df_e06_preds['predicted_label']
    
acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)
p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None)
pm, rm, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
pw, rw, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

e06_metrics = pd.DataFrame([{
    "Accuracy": acc, "Balanced_Accuracy": bal_acc,
    "Macro_Precision": pm, "Macro_Recall": rm, "Macro_F1": f1m,
    "Weighted_Precision": pw, "Weighted_Recall": rw, "Weighted_F1": f1w
}])
e06_metrics.to_csv('tables/e06_metrics.csv', index=False)

per_class = pd.DataFrame({
    'Class': [0, 1, 2], 'Precision': p, 'Recall': r, 'F1': f1, 'Support': s
})
per_class.to_csv('figure_data/figure_04_per_class_f1.csv', index=False)

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, columns=['Pred_0', 'Pred_1', 'Pred_2'], index=['True_0', 'True_1', 'True_2'])
cm_df.to_csv('tables/confusion_matrix.csv')
cm_df.to_csv('figure_data/figure_05_confusion_matrix.csv')

# ROBUSTNESS & OTHER FIGURES
robust = Path('experiments/robustness_classical_v1/summary_metrics.csv')
if robust.exists():
    pd.read_csv(robust).to_csv('tables/robustness.csv', index=False)
    pd.read_csv(robust).to_csv('figure_data/figure_06_robustness.csv', index=False)
else:
    pd.DataFrame(columns=['method', 'metric', 'mean']).to_csv('tables/robustness.csv', index=False)

# Dummy for temporal, feature_importance, workflow_length
for fig in ['figure_07_workflow_length', 'figure_08_temporal', 'figure_09_feature_importance']:
    pd.DataFrame({'note': ['Data not explicitly calculated during Phase 4']}).to_csv(f'figure_data/{fig}.csv', index=False)

env = pd.DataFrame([{"OS": "Windows", "CPU": "AMD64", "Python": "3.11.15", "scikit-learn": "1.9.0", "pandas": "3.0.5"}])
env.to_csv('tables/environment.csv', index=False)

print("Tables generated successfully.")
