import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

out_dir = 'reports/figures/phase_4_4'
os.makedirs(out_dir, exist_ok=True)

report_lines = []
report_lines.append("### E06 Metric Reference")
report_lines.append("Verified historical E06 Validation Macro-F1 = 0.4857 and Test Macro-F1 = 0.4972.\n")

# A. Ablation Matrix
models = {
    'E02': 'e02_tfidf_linear_svm',
    'E05': 'e05_structure_gbdt',
    'E06': 'e06_hybrid_logreg',
    'E08': 'e08_codebert_text',
    'E09': 'e09_structural_mlp',
    'E10': 'e10_neural_hybrid',
    'E11': 'e11_neural_hybrid_finetuned'
}
metrics = {}
for m_id, exp_dir in models.items():
    with open(f'experiments/{exp_dir}/metrics.json', 'r') as f:
        m_data = json.load(f)
        if 'val_macro_f1' in m_data:
            # Flattened like E08
            metrics[m_id] = {
                'val_macro_f1': m_data['val_macro_f1'],
                'test_macro_f1': m_data['test_macro_f1'],
                'test_accuracy': m_data['test_accuracy'],
                'balanced_accuracy': m_data['test_balanced_accuracy'],
                'class_0_f1': m_data.get('test_class_0_f1', 0),
                'class_1_f1': m_data.get('test_class_1_f1', 0),
                'class_2_f1': m_data.get('test_class_2_f1', 0)
            }
        else:
            val_key = 'validation' if 'validation' in m_data else 'val'
            metrics[m_id] = {
                'val_macro_f1': m_data[val_key]['macro_f1'],
                'test_macro_f1': m_data['test']['macro_f1'],
                'test_accuracy': m_data['test']['accuracy'],
                'balanced_accuracy': m_data['test']['balanced_accuracy'],
                'class_0_f1': m_data['test'].get('class_0_f1', m_data['test'].get('per_class_f1', {}).get('0', 0)),
                'class_1_f1': m_data['test'].get('class_1_f1', m_data['test'].get('per_class_f1', {}).get('1', 0)),
                'class_2_f1': m_data['test'].get('class_2_f1', m_data['test'].get('per_class_f1', {}).get('2', 0))
            }

report_lines.append("### Ablation Matrix")
report_lines.append("| Model | View | Val Macro-F1 | Test Macro-F1 | Test Acc | Balanced Acc |")
report_lines.append("|---|---|---|---|---|---|")
report_lines.append(f"| E02 TF-IDF SVM | Text | {metrics['E02']['val_macro_f1']:.4f} | {metrics['E02']['test_macro_f1']:.4f} | {metrics['E02']['test_accuracy']:.4f} | {metrics['E02']['balanced_accuracy']:.4f} |")
report_lines.append(f"| E05 Struct GBDT | Struct | {metrics['E05']['val_macro_f1']:.4f} | {metrics['E05']['test_macro_f1']:.4f} | {metrics['E05']['test_accuracy']:.4f} | {metrics['E05']['balanced_accuracy']:.4f} |")
report_lines.append(f"| E06 Hybrid LogReg | Text+Struct | {metrics['E06']['val_macro_f1']:.4f} | {metrics['E06']['test_macro_f1']:.4f} | {metrics['E06']['test_accuracy']:.4f} | {metrics['E06']['balanced_accuracy']:.4f} |")
report_lines.append(f"| E08 CodeBERT | Text | {metrics['E08']['val_macro_f1']:.4f} | {metrics['E08']['test_macro_f1']:.4f} | {metrics['E08']['test_accuracy']:.4f} | {metrics['E08']['balanced_accuracy']:.4f} |")
report_lines.append(f"| E09 Struct MLP | Struct | {metrics['E09']['val_macro_f1']:.4f} | {metrics['E09']['test_macro_f1']:.4f} | {metrics['E09']['test_accuracy']:.4f} | {metrics['E09']['balanced_accuracy']:.4f} |")
report_lines.append(f"| E10 Neural Hybrid (Frozen) | Text+Struct | {metrics['E10']['val_macro_f1']:.4f} | {metrics['E10']['test_macro_f1']:.4f} | {metrics['E10']['test_accuracy']:.4f} | {metrics['E10']['balanced_accuracy']:.4f} |")
report_lines.append(f"| E11 Neural Hybrid (Trained) | Text+Struct | {metrics['E11']['val_macro_f1']:.4f} | {metrics['E11']['test_macro_f1']:.4f} | {metrics['E11']['test_accuracy']:.4f} | {metrics['E11']['balanced_accuracy']:.4f} |\n")

# B. Incremental Gains
report_lines.append("### Incremental Gains")
def d(m1, m2):
    return (metrics[m1]['val_macro_f1'] - metrics[m2]['val_macro_f1'], metrics[m1]['test_macro_f1'] - metrics[m2]['test_macro_f1'])

v_s_t, t_s_t = d('E06', 'E02')
v_t_s, t_t_s = d('E06', 'E05')
v_nh_cb, t_nh_cb = d('E10', 'E08')
v_nh_smlp, t_nh_smlp = d('E10', 'E09')
v_ft, t_ft = d('E11', 'E10')

report_lines.append(f"- Structure gain over text (E06 - E02): Val {v_s_t:+.4f} | Test {t_s_t:+.4f}")
report_lines.append(f"- Text gain over structure (E06 - E05): Val {v_t_s:+.4f} | Test {t_t_s:+.4f}")
report_lines.append(f"- Neural hybrid gain over CodeBERT (E10 - E08): Val {v_nh_cb:+.4f} | Test {t_nh_cb:+.4f}")
report_lines.append(f"- Neural hybrid gain over structural MLP (E10 - E09): Val {v_nh_smlp:+.4f} | Test {t_nh_smlp:+.4f}")
report_lines.append(f"- Fine-tuning gain (E11 - E10): Val {v_ft:+.4f} | Test {t_ft:+.4f}\n")

# C. Per-Class Results
report_lines.append("### Per-Class Results")
c0_scores = {m: metrics[m]['class_0_f1'] for m in models}
c1_scores = {m: metrics[m]['class_1_f1'] for m in models}
c2_scores = {m: metrics[m]['class_2_f1'] for m in models}

report_lines.append("| Model | Class 0 F1 | Class 1 F1 | Class 2 F1 |")
report_lines.append("|---|---|---|---|")
for m in models:
    report_lines.append(f"| {m} | {c0_scores[m]:.4f} | {c1_scores[m]:.4f} | {c2_scores[m]:.4f} |")

avg_c0 = np.mean(list(c0_scores.values()))
avg_c1 = np.mean(list(c1_scores.values()))
avg_c2 = np.mean(list(c2_scores.values()))

easiest = 0 if avg_c0 > max(avg_c1, avg_c2) else 1 if avg_c1 > max(avg_c0, avg_c2) else 2
hardest = 0 if avg_c0 < min(avg_c1, avg_c2) else 1 if avg_c1 < min(avg_c0, avg_c2) else 2

report_lines.append(f"\n- **Easiest class:** {easiest}")
report_lines.append(f"- **Hardest class:** {hardest}")
report_lines.append(f"- **Model with best Class 0 F1:** {max(c0_scores, key=c0_scores.get)}")
report_lines.append(f"- **Model with best Class 1 F1:** {max(c1_scores, key=c1_scores.get)}")
report_lines.append(f"- **Model with best Class 2 F1:** {max(c2_scores, key=c2_scores.get)}\n")

# D. Confusion Analysis
report_lines.append("### Confusion Analysis")
def conf_analysis(model_name, cm_path):
    if 'test_confusion_matrix' in cm_path:
        cm_mat = pd.read_csv(cm_path, header=None).values
    else:
        cm = pd.read_csv(cm_path, index_col=0)
        cm_mat = cm.values
    report_lines.append(f"#### {model_name}")
    for i in range(3):
        tp = cm_mat[i,i]
        fp = cm_mat[:,i].sum() - tp
        fn = cm_mat[i,:].sum() - tp
        report_lines.append(f"- Class {i}: TP={tp}, FP={fp}, FN={fn}")
    total = cm_mat.sum()
    report_lines.append("Dominant Confusions:")
    for i in range(3):
        for j in range(3):
            if i != j:
                perc = cm_mat[i,j] / total * 100
                if perc > 3.0: # threshold for dominance
                    report_lines.append(f"- True {i} -> Pred {j}: {cm_mat[i,j]} ({perc:.1f}%)")

conf_analysis("E06", 'experiments/e06_hybrid_logreg/confusion_matrix.csv')
conf_analysis("E10", 'experiments/e10_neural_hybrid/test_confusion_matrix.csv')
report_lines.append("")

# E. Strict Test
report_lines.append("### Strict Novel-Configuration Results")
df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
df_text = pd.read_parquet('data/processed/model_ready_text_v1.parquet')

train_texts = set(df_text[df_text['split'] == 'train']['normalized_workflow_text'])
test_df = df_text[df_text['split'] == 'test']
test_sids = test_df['sample_id'].values
test_labels = test_df['final_label'].values

is_strict = ~test_df['normalized_workflow_text'].isin(train_texts)

def strict_eval(exp_dir):
    p_path = f"experiments/{exp_dir}/test_predictions.parquet"
    if not os.path.exists(p_path): return None, None
    preds_df = pd.read_parquet(p_path)
    # sort to match test_sids order
    preds_df = preds_df.set_index('sample_id').loc[test_sids]
    preds = preds_df['pred_label'].values
    
    normal_f1 = f1_score(test_labels, preds, average='macro')
    strict_f1 = f1_score(test_labels[is_strict], preds[is_strict], average='macro')
    return normal_f1, strict_f1

n_e02, s_e02 = strict_eval('e02_tfidf_linear_svm')
n_e06, s_e06 = strict_eval('e06_hybrid_logreg')
n_e10, s_e10 = strict_eval('e10_neural_hybrid')

report_lines.append(f"- Original test size: {len(test_labels)}")
report_lines.append(f"- Strict test size: {is_strict.sum()}")
report_lines.append(f"- Removed rows: {len(test_labels) - is_strict.sum()}\n")
report_lines.append("| Model | Normal Macro-F1 | Strict Macro-F1 | Delta |")
report_lines.append("|---|---|---|---|")
if n_e02: report_lines.append(f"| E02 | {n_e02:.4f} | {s_e02:.4f} | {s_e02-n_e02:+.4f} |")
if n_e06: report_lines.append(f"| E06 | {n_e06:.4f} | {s_e06:.4f} | {s_e06-n_e06:+.4f} |")
if n_e10: report_lines.append(f"| E10 | {n_e10:.4f} | {s_e10:.4f} | {s_e10-n_e10:+.4f} |\n")

# F. Repo Robustness
report_lines.append("### Repository-Size Robustness")
repo_counts = df_prov['repository'].value_counts()
test_prov = df_prov[df_prov['split'] == 'test'].copy()
test_prov['repo_size'] = test_prov['repository'].map(repo_counts)

def bin_repo(s):
    if s == 1: return "1 version"
    elif s <= 5: return "2-5 versions"
    elif s <= 20: return "6-20 versions"
    else: return ">20 versions"

test_prov['repo_group'] = test_prov['repo_size'].apply(bin_repo)

if os.path.exists('experiments/e06_hybrid_logreg/test_predictions.parquet'):
    p_df = pd.read_parquet('experiments/e06_hybrid_logreg/test_predictions.parquet')
    test_prov = test_prov.merge(p_df, on='sample_id', how='left')
    
    report_lines.append("| Group | Num Repos | Num Samples | Accuracy | Macro-F1 |")
    report_lines.append("|---|---|---|---|---|")
    for grp in ["1 version", "2-5 versions", "6-20 versions", ">20 versions"]:
        sub = test_prov[test_prov['repo_group'] == grp]
        if len(sub) == 0: continue
        acc = accuracy_score(sub['true_label'], sub['pred_label'])
        f1 = f1_score(sub['true_label'], sub['pred_label'], average='macro')
        n_repos = sub['repository'].nunique()
        report_lines.append(f"| {grp} | {n_repos} | {len(sub)} | {acc:.4f} | {f1:.4f} |")
report_lines.append("")

# Write down script to a temp file and execute it (it's getting long)
with open('scripts/analyze_phase_4_4_part1.txt', 'w') as f:
    f.write('\n'.join(report_lines))
