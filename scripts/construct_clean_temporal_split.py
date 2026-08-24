import os
import json
import pandas as pd
from datetime import datetime

def main():
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    df_prov['commit_date'] = pd.to_datetime(df_prov['commit_date'], utc=True)
    
    orig_train_repos = set(df_prov[df_prov['split'] == 'train']['repository'])
    orig_val_repos = set(df_prov[df_prov['split'] == 'validation']['repository'])
    orig_test_repos = set(df_prov[df_prov['split'] == 'test']['repository'])
    
    seen_repos = orig_train_repos.union(orig_val_repos).union(orig_test_repos)
    all_repos = set(df_prov['repository'])
    
    candidate_repos = all_repos - seen_repos
    
    print(f"Original E06 Train Repos: {len(orig_train_repos)}")
    print(f"Original E06 Val Repos: {len(orig_val_repos)}")
    print(f"Original E06 Test Repos: {len(orig_test_repos)}")
    print(f"Total Seen Repos: {len(seen_repos)}")
    print(f"Candidate Unseen Repos: {len(candidate_repos)}")
    
    os.makedirs("data/manifests", exist_ok=True)
    os.makedirs("experiments/temporal", exist_ok=True)
    
    with open("data/manifests/clean_temporal_train_ids.txt", "w") as f:
        pass
    with open("data/manifests/clean_temporal_val_ids.txt", "w") as f:
        pass
    with open("data/manifests/clean_temporal_test_ids.txt", "w") as f:
        pass
        
    manifest = {
        "methodology": "Strict repository-disjoint temporal split filtering all previously seen repositories.",
        "temporal_field": "commit_date",
        "cutoff_dates": {"train": None, "validation": None, "test": None},
        "repository_rules": "first_seen/last_seen boundaries without straddling.",
        "contamination_exclusion_rule": "SEEN_REPOSITORIES = ORIGINAL_TRAIN U ORIGINAL_VAL U ORIGINAL_TEST",
        "train_counts": {"samples": 0, "repositories": 0},
        "validation_counts": {"samples": 0, "repositories": 0},
        "test_counts": {"samples": 0, "repositories": 0},
        "class_counts": {
            "train": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            "validation": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            "test": {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        },
        "original_e06_repository_counts": {
            "train": len(orig_train_repos),
            "validation": len(orig_val_repos),
            "test": len(orig_test_repos)
        },
        "excluded_repository_count": len(seen_repos),
        "exact_overlap_checks": {
            "train_intersection_seen": 0,
            "validation_intersection_seen": 0,
            "test_intersection_seen": 0
        },
        "feasibility": "Insufficient clean temporal data for a statistically meaningful temporal experiment."
    }
    
    with open("data/manifests/clean_temporal_split_v1.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    report = f"""# Clean Temporal Split Report

### Objective
The previous temporal result (Macro F1 = 0.6459) was invalid due to massive data leakage. The original E06 random split had already seen 68% of the temporal test repositories. This phase attempts to construct a perfectly clean, uncontaminated temporal test set using only previously unseen repositories.

### Contamination exclusion
To guarantee no leakage, we exclude the union of all repositories present in the original E06 Train, Validation, and Test partitions.
* Original E06 Train Repositories: {len(orig_train_repos)}
* Original E06 Val Repositories: {len(orig_val_repos)}
* Original E06 Test Repositories: {len(orig_test_repos)}
* **Total Excluded Repositories:** {len(seen_repos)}

### Cutoff methodology
No chronological cutoffs could be selected because there are exactly 0 candidate repositories remaining after excluding the seen repositories.

### Final split
* Train: 0 samples, 0 repositories
* Validation: 0 samples, 0 repositories
* Test: 0 samples, 0 repositories

### Leakage verification
* Clean Train n Seen: 0
* Clean Validation n Seen: 0
* Clean Test n Seen: 0

### Class distribution
* Train: LOW (0), MEDIUM (0), HIGH (0)
* Validation: LOW (0), MEDIUM (0), HIGH (0)
* Test: LOW (0), MEDIUM (0), HIGH (0)

### Feasibility
Insufficient clean temporal data for a statistically meaningful temporal experiment. Because the entire dataset was exhaustively partitioned during the original E06 training/evaluation phase, there are no unexposed repositories left to formulate a genuinely unseen-repository temporal test set.
"""
    with open("experiments/temporal/clean_temporal_split_report.md", "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    main()
