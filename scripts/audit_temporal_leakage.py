import pandas as pd
import json

def main():
    print("Loading data...")
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    
    # 1. Original E06 Split
    df = df.merge(df_prov[['sample_id', 'repository', 'commit_date', 'workflow_id', 'workflow_path', 'commit_sha']], on='sample_id', how='left')
    df['commit_date'] = pd.to_datetime(df['commit_date'], utc=True)
    
    orig_train = df[df['split'] == 'train']
    orig_val = df[df['split'] == 'validation']
    orig_test = df[df['split'] == 'test']
    
    orig_train_repos = set(orig_train['repository'])
    orig_val_repos = set(orig_val['repository'])
    orig_test_repos = set(orig_test['repository'])
    
    print(f"Original E06 Train Repos: {len(orig_train_repos)}")
    print(f"Original E06 Val Repos: {len(orig_val_repos)}")
    print(f"Original E06 Test Repos: {len(orig_test_repos)}")
    
    # 2. Temporal Split
    temporal_field = 'commit_date'
    t1 = df[temporal_field].quantile(0.85)
    t2 = df[temporal_field].quantile(0.95)
    
    repos = df.groupby('repository')[temporal_field].agg(['min', 'max']).reset_index()
    temp_train_repos = set(repos[repos['max'] <= t1]['repository'])
    temp_val_repos = set(repos[(repos['min'] > t1) & (repos['max'] <= t2)]['repository'])
    temp_test_repos = set(repos[repos['min'] > t2]['repository'])
    
    temp_train = df[df['repository'].isin(temp_train_repos)]
    temp_val = df[df['repository'].isin(temp_val_repos)]
    temp_test = df[df['repository'].isin(temp_test_repos)]
    
    print(f"Temporal Train Repos: {len(temp_train_repos)}")
    print(f"Temporal Val Repos: {len(temp_val_repos)}")
    print(f"Temporal Test Repos: {len(temp_test_repos)}")
    
    # 3. Critical Overlap Check
    tr_overlap = orig_train_repos.intersection(temp_test_repos)
    va_overlap = orig_val_repos.intersection(temp_test_repos)
    te_overlap = orig_test_repos.intersection(temp_test_repos)
    
    tr_pct = len(tr_overlap) / len(temp_test_repos) * 100
    va_pct = len(va_overlap) / len(temp_test_repos) * 100
    te_pct = len(te_overlap) / len(temp_test_repos) * 100
    
    print(f"Orig Train n Temp Test: {len(tr_overlap)} ({tr_pct:.2f}%)")
    print(f"Orig Val n Temp Test: {len(va_overlap)} ({va_pct:.2f}%)")
    print(f"Orig Test n Temp Test: {len(te_overlap)} ({te_pct:.2f}%)")
    
    # 4. Workflow Version Overlap
    # Filter datasets to just the overlapping repositories
    orig_train_overlap = orig_train[orig_train['repository'].isin(tr_overlap)]
    temp_test_overlap = temp_test[temp_test['repository'].isin(tr_overlap)]
    
    # Exact sample overlap (sample_id)
    exact_samples_train = set(orig_train_overlap['sample_id'])
    exact_samples_test = set(temp_test_overlap['sample_id'])
    exact_sample_overlap = exact_samples_train.intersection(exact_samples_test)
    
    # Same workflow identity (workflow_id or workflow_path within the same repo)
    wf_ids_train = set(orig_train_overlap['workflow_id'].dropna().astype(str))
    wf_ids_test = set(temp_test_overlap['workflow_id'].dropna().astype(str))
    
    # Let's create a combined identity: repo + workflow_path to see if the SAME workflow file was seen
    orig_train_overlap_id = set(orig_train_overlap['repository'] + ":" + orig_train_overlap['workflow_path'])
    temp_test_overlap_id = set(temp_test_overlap['repository'] + ":" + temp_test_overlap['workflow_path'])
    same_workflow_overlap = orig_train_overlap_id.intersection(temp_test_overlap_id)
    
    print(f"Exact sample overlap: {len(exact_sample_overlap)}")
    print(f"Same workflow identity overlap: {len(same_workflow_overlap)}")
    
    # Generate Report
    report = f"""# Temporal Leakage Audit Report

### Original E06 Split
* **Train repositories:** {len(orig_train_repos)}
* **Validation repositories:** {len(orig_val_repos)}
* **Test repositories:** {len(orig_test_repos)}

### Temporal Split
* **Train repositories:** {len(temp_train_repos)}
* **Validation repositories:** {len(temp_val_repos)}
* **Test repositories:** {len(temp_test_repos)}

### Cross-Split Overlaps

| Comparison | Overlap repositories | Percentage |
|---|---:|---:|
| Original E06 Train n Temporal Test | {len(tr_overlap)} | {tr_pct:.2f}% |
| Original E06 Validation n Temporal Test | {len(va_overlap)} | {va_pct:.2f}% |
| Original E06 Test n Temporal Test | {len(te_overlap)} | {te_pct:.2f}% |

### Workflow-Level Overlap (Original E06 Train vs Temporal Test)
* **Overlapping repositories:** {len(tr_overlap)}
* **Overlapping workflow versions (exact sample matches):** {len(exact_sample_overlap)}
* **Same-workflow-identity matches (same repository + workflow path):** {len(same_workflow_overlap)}

### Analysis
"""

    if len(tr_overlap) == 0:
        report += "CASE A: No repository overlap detected between Original E06 Train and Temporal Test. The temporal evaluation is considered much stronger evidence of out-of-time generalization.\n"
    else:
        report += "CASE B: Repository overlap detected. The temporal evaluation must be considered potentially contaminated. Do NOT call the 0.6459 result a valid unseen-repository temporal generalization result.\n"
        
    if len(exact_sample_overlap) > 0:
        report += "\nCASE C: Exact workflow versions overlap. The temporal evaluation is strictly invalid due to data leakage.\n"

    with open("experiments/temporal/temporal_leakage_audit.md", "w", encoding='utf-8') as f:
        f.write(report)
        
    results = {
        "orig_train_repos": len(orig_train_repos),
        "temp_test_repos": len(temp_test_repos),
        "overlap_repos": len(tr_overlap),
        "overlap_pct": tr_pct,
        "exact_samples": len(exact_sample_overlap),
        "same_workflow": len(same_workflow_overlap)
    }
    with open("experiments/temporal/audit_summary.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
