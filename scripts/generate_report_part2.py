import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


with open('scripts/analyze_phase_4_4_part1.txt', 'r') as f:
    report_lines = f.read().splitlines()

df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
df_text = pd.read_parquet('data/processed/model_ready_text_v1.parquet')

# Load E06 preds
e06_preds = pd.read_parquet('experiments/e06_hybrid_logreg/test_predictions.parquet')

# G. Temporal Robustness
report_lines.append("### Temporal Robustness")
test_prov = df_prov[df_prov['split'] == 'test'].copy()
test_prov = test_prov.merge(e06_preds, on='sample_id')
test_prov['commit_date'] = pd.to_datetime(test_prov['commit_date'])

train_prov = df_prov[df_prov['split'] == 'train'].copy()
train_prov['commit_date'] = pd.to_datetime(train_prov['commit_date'])

# We don't need quantiles from train for temporal if it's diagnostic, but let's just bin into quarters based on TEST itself to see.
test_prov['temp_bin'] = pd.qcut(test_prov['commit_date'], q=4, labels=['Q1 (Oldest)', 'Q2', 'Q3', 'Q4 (Newest)'])

report_lines.append("| Temporal Bin | Num Samples | Macro-F1 |")
report_lines.append("|---|---|---|")
for grp in ['Q1 (Oldest)', 'Q2', 'Q3', 'Q4 (Newest)']:
    sub = test_prov[test_prov['temp_bin'] == grp]
    if len(sub) == 0: continue
    f1 = f1_score(sub['true_label'], sub['pred_label'], average='macro')
    report_lines.append(f"| {grp} | {len(sub)} | {f1:.4f} |")
report_lines.append("")

# H. Workflow-Length Robustness
report_lines.append("### Workflow-Length Robustness")
train_text = df_text[df_text['split'] == 'train'].copy()
test_text = df_text[df_text['split'] == 'test'].copy()

train_text['len'] = train_text['normalized_workflow_text'].str.len()
test_text['len'] = test_text['normalized_workflow_text'].str.len()

bins = [0] + list(train_text['len'].quantile([0.25, 0.5, 0.75]).values) + [np.inf]
labels = ['SHORT', 'MEDIUM', 'LONG', 'VERY_LONG']
test_text['len_bin'] = pd.cut(test_text['len'], bins=bins, labels=labels)

test_text = test_text.merge(e06_preds, on='sample_id')
report_lines.append("| Length Group | Num Samples | Macro-F1 |")
report_lines.append("|---|---|---|")
for grp in labels:
    sub = test_text[test_text['len_bin'] == grp]
    if len(sub) == 0: continue
    f1 = f1_score(sub['true_label'], sub['pred_label'], average='macro')
    report_lines.append(f"| {grp} | {len(sub)} | {f1:.4f} |")
report_lines.append("")

# I. CodeBERT Truncation Diagnostic
report_lines.append("### CodeBERT Truncation Analysis")
from datasets import load_from_disk
ds_text = load_from_disk('data/intermediate/codebert_tokenized/512')
test_ds = ds_text.filter(lambda x: x['split'] == 'test')
mask = np.array([sum(x) for x in test_ds['attention_mask']])
trunc = (mask == 512)

def cb_diag(model_id):
    p_path = f"experiments/{model_id}/test_predictions.parquet"
    if not os.path.exists(p_path): return
    p_df = pd.read_parquet(p_path)
    # sort by sample_id in order of test_ds
    sids = test_ds['sample_id']
    p_df = p_df.set_index('sample_id').loc[sids]
    
    f1_short = f1_score(p_df['true_label'][~trunc], p_df['pred_label'][~trunc], average='macro')
    f1_long = f1_score(p_df['true_label'][trunc], p_df['pred_label'][trunc], average='macro')
    
    report_lines.append(f"#### {model_id}")
    report_lines.append(f"- <512 tokens: {sum(~trunc)} samples ({(sum(~trunc)/len(trunc))*100:.1f}%) | Macro-F1: {f1_short:.4f}")
    report_lines.append(f"- >=512 tokens: {sum(trunc)} samples ({(sum(trunc)/len(trunc))*100:.1f}%) | Macro-F1: {f1_long:.4f}\n")

cb_diag('e08_codebert_text')
cb_diag('e10_neural_hybrid')

# J. Structural Feature Importance
report_lines.append("### Structural Feature Importance")
feat_imp_path = 'experiments/e04_structure_rf/feature_importance.csv'
if os.path.exists(feat_imp_path):
    fi = pd.read_csv(feat_imp_path)
    top20 = fi.head(20)
    report_lines.append("| Feature | Importance |")
    report_lines.append("|---|---|")
    for _, r in top20.iterrows():
        report_lines.append(f"| {r['feature']} | {r['importance']:.4f} |")
report_lines.append("")

# K. Error Analysis
report_lines.append("### Error Analysis (E06)")
errors = e06_preds[e06_preds['true_label'] != e06_preds['pred_label']]
report_lines.append(f"Total errors: {len(errors)} / {len(e06_preds)}")

samp = errors.sample(n=min(20, len(errors)), random_state=42)
samp = samp.merge(df_text, on='sample_id')
for idx, r in samp.iterrows():
    report_lines.append(f"**Sample:** {r['sample_id']} (True: {r['true_label']} | Pred: {r['pred_label']})")
    text = r['normalized_workflow_text']
    report_lines.append(f"`yaml\n{text[:200]}...\n`\n")

# L. Calibration
report_lines.append("### Calibration / Confidence")
report_lines.append("E06 (Logistic Regression) provides probabilistic output, but this was not logged for this evaluation.\n")

# M. Paired Comparison
report_lines.append("### Paired Comparison")
try:
    p2 = pd.read_parquet('experiments/e02_tfidf_linear_svm/test_predictions.parquet').set_index('sample_id')
    p5 = pd.read_parquet('experiments/e05_structure_gbdt/test_predictions.parquet').set_index('sample_id')
    p6 = e06_preds.set_index('sample_id')
    
    import scipy.stats
    
    # E06 vs E02
    both_correct_2 = (p6['true_label'] == p6['pred_label']) & (p2['true_label'] == p2['pred_label'])
    e6_right_e2_wrong = (p6['true_label'] == p6['pred_label']) & (p2['true_label'] != p2['pred_label'])
    e2_right_e6_wrong = (p6['true_label'] != p6['pred_label']) & (p2['true_label'] == p2['pred_label'])
    both_wrong_2 = (p6['true_label'] != p6['pred_label']) & (p2['true_label'] != p2['pred_label'])
    
    b2 = e6_right_e2_wrong.sum()
    c2 = e2_right_e6_wrong.sum()
    stat_2 = (abs(b2 - c2) - 1)**2 / (b2 + c2) if (b2+c2) > 0 else 0
    p_val_2 = scipy.stats.chi2.sf(stat_2, 1)
    
    report_lines.append("#### E06 (Hybrid) vs E02 (Text Only)")
    report_lines.append(f"- E06 Wins: {b2}")
    report_lines.append(f"- E06 Loses: {c2}")
    report_lines.append(f"- Tied Correct: {both_correct_2.sum()}")
    report_lines.append(f"- Tied Incorrect: {both_wrong_2.sum()}")
    report_lines.append(f"- McNemar p-value: {p_val_2:.4f}")
    
    # E06 vs E05
    both_correct_5 = (p6['true_label'] == p6['pred_label']) & (p5['true_label'] == p5['pred_label'])
    e6_right_e5_wrong = (p6['true_label'] == p6['pred_label']) & (p5['true_label'] != p5['pred_label'])
    e5_right_e6_wrong = (p6['true_label'] != p6['pred_label']) & (p5['true_label'] == p5['pred_label'])
    both_wrong_5 = (p6['true_label'] != p6['pred_label']) & (p5['true_label'] != p5['pred_label'])
    
    b5 = e6_right_e5_wrong.sum()
    c5 = e5_right_e6_wrong.sum()
    stat_5 = (abs(b5 - c5) - 1)**2 / (b5 + c5) if (b5+c5) > 0 else 0
    p_val_5 = scipy.stats.chi2.sf(stat_5, 1)
    
    report_lines.append("\n#### E06 (Hybrid) vs E05 (Structure Only)")
    report_lines.append(f"- E06 Wins: {b5}")
    report_lines.append(f"- E06 Loses: {c5}")
    report_lines.append(f"- Tied Correct: {both_correct_5.sum()}")
    report_lines.append(f"- Tied Incorrect: {both_wrong_5.sum()}")
    report_lines.append(f"- McNemar p-value: {p_val_5:.4f}\n")
except Exception as e:
    report_lines.append(f"Could not compute paired comparison: {str(e)}\n")

# N. Final Model Candidate
report_lines.append("### Final Model Candidate")
report_lines.append("E06 (Hybrid Logistic Regression) is the final candidate. It achieves the highest Macro-F1 across both Validation and Test splits.")

with open('reports/phase_4_4_ablation_report.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
