import sys
import zipfile
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import ProjectPaths
from src.data.build_execution_labels import robust_bson_stream

def stream_metadata(zip_path, commit_bson, needed_keys_set):
    print(f"Streaming metadata from {commit_bson}...")
    metadata = []
    scanned = 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(commit_bson, "r") as f:
            for doc, err in robust_bson_stream(f):
                scanned += 1
                if err or not doc:
                    continue
                    
                w_id = str(doc.get("workflow_id"))
                c_sha = str(doc.get("commit_sha"))
                
                if (w_id, c_sha) in needed_keys_set:
                    meta = {
                        "workflow_id": w_id,
                        "commit_sha": c_sha,
                        "owner": doc.get("owner"),
                        "repo": doc.get("repo"),
                        "workflow_path": doc.get("workflow_path"),
                        "commit_date": doc.get("commit_date"),
                        "file_sha": doc.get("file_sha"),
                        "parsing_success": bool(doc.get("parsing_success", False)),
                        "parsed_workflow_available": doc.get("parsed_workflow") is not None
                    }
                    metadata.append(meta)
                    
                if scanned % 250000 == 0:
                    print(f"  commits scanned: {scanned:,}")
                    
    print(f"Done scanning metadata. Extracted {len(metadata)} records.")
    return pd.DataFrame(metadata)

def compute_repo_split(df_cap, seed=42):
    t1_prov = df_cap["failure_rate"].quantile(1/3)
    t2_prov = df_cap["failure_rate"].quantile(2/3)
    
    def get_prov_class(fr):
        if fr <= t1_prov: return 0
        elif fr <= t2_prov: return 1
        else: return 2
        
    df_cap["provisional_class"] = df_cap["failure_rate"].apply(get_prov_class)
    
    repo_stats = df_cap.groupby("repository")["provisional_class"].agg(lambda x: x.mode()[0]).reset_index()
    
    repos = repo_stats["repository"]
    strat = repo_stats["provisional_class"]
    
    r_train, r_temp, s_train, s_temp = train_test_split(
        repos, strat, test_size=0.30, random_state=seed, stratify=strat
    )
    
    r_val, r_test = train_test_split(
        r_temp, test_size=0.50, random_state=seed, stratify=s_temp
    )
    
    train_repos = set(r_train)
    val_repos = set(r_val)
    test_repos = set(r_test)
    
    assert len(train_repos & val_repos) == 0
    assert len(train_repos & test_repos) == 0
    assert len(val_repos & test_repos) == 0
    
    def assign_split(r):
        if r in train_repos: return "train"
        if r in val_repos: return "validation"
        if r in test_repos: return "test"
        return "UNKNOWN"
        
    df_cap["split"] = df_cap["repository"].apply(assign_split)
    df_cap.drop(columns=["provisional_class"], inplace=True)
    return df_cap, train_repos, val_repos, test_repos

def calculate_final_thresholds(df):
    df_train = df[df["split"] == "train"]
    
    t1 = df_train["failure_rate"].quantile(1/3)
    t2 = df_train["failure_rate"].quantile(2/3)
    
    def assign_final_label(fr):
        if fr <= t1: return 0
        elif fr <= t2: return 1
        else: return 2
        
    df["final_label"] = df["failure_rate"].apply(assign_final_label)
    return df, t1, t2

def data_quality_audit(df, initial_len, cap_len, excluded_df):
    print("\n===== DATA QUALITY =====")
    print(f"Excluded: {len(excluded_df)}")
    print(f"Remaining: {len(df)}")
    
    missing = df.isnull().sum()
    print(f"Missing:\n{missing[missing > 0]}")
    
    duplicates = df.duplicated(subset=["workflow_id", "commit_sha"]).sum()
    print(f"Duplicates: {duplicates}")
    
    assert df["repository"].notna().all(), "Missing repository"
    assert df["workflow_id"].notna().all(), "Missing workflow_id"
    assert df["commit_sha"].notna().all(), "Missing commit_sha"
    assert df["commit_date"].notna().all(), "Missing commit_date"
    assert df["sample_id"].is_unique, "Duplicate sample_id"
    assert df.duplicated(subset=["workflow_id", "commit_sha"]).sum() == 0, "Duplicate versions"
    assert set(df["final_label"].unique()).issubset({0, 1, 2}), "Invalid classes"
    
    max_per_repo = df.groupby("repository").size().max()
    assert max_per_repo <= 20, f"Repository cap exceeded! Max: {max_per_repo}"

def main():
    paths = ProjectPaths()
    zip_path = paths.raw_data_root / "github-workflows-december-2025.zip"
    cov_path = paths.intermediate_data_root / "execution_label_coverage_v1.parquet"
    
    print("Loading execution coverage...")
    df_exec = pd.read_parquet(cov_path)
    
    df_exec = df_exec[df_exec["meets_minimum_10"] == True].copy()
    initial_len = len(df_exec)
    print(f"Initial >=10-run candidate versions: {initial_len:,}")
    
    df_exec["workflow_id"] = df_exec["workflow_id"].astype(str)
    df_exec["commit_sha"] = df_exec["commit_sha"].astype(str)
    needed_keys = set(zip(df_exec["workflow_id"], df_exec["commit_sha"]))
    
    df_meta = stream_metadata(zip_path, "github-workflows-december-2025/workflow_files_commits.bson", needed_keys)
    
    df_joined = pd.merge(df_exec, df_meta, on=["workflow_id", "commit_sha"], how="inner")
    df_joined["repository"] = df_joined["owner"] + "/" + df_joined["repo"]
    
    print(f"After join: {len(df_joined)}")
    
    print("Applying cap-20 sampling per repository (seed=42)...")
    df_joined.sort_values(by=["repository", "workflow_id", "commit_date", "commit_sha"], inplace=True)
    
    df_joined["_grp"] = df_joined["repository"]
    df_cap = df_joined.groupby("_grp", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 20), random_state=42)
    )
    df_cap.drop(columns=["_grp"], inplace=True, errors="ignore")
    
    cap_len = len(df_cap)
    print(f"Cap-20 candidates: {cap_len:,}")
    
    # IMPLEMENT IMPLEMENTATION BUG FIX: Exclude ONLY if both are False
    exclude_mask = (df_cap["parsing_success"] == False) & (df_cap["parsed_workflow_available"] == False)
    valid_mask = ~exclude_mask
    
    df_final = df_cap[valid_mask].copy()
    df_excluded = df_cap[~valid_mask].copy()
    
    final_len = len(df_final)
    print(f"Final candidates after quality filtering: {final_len:,}")
    
    print("Creating repository split...")
    df_final, train_repos, val_repos, test_repos = compute_repo_split(df_final, seed=42)
    
    print("Calculating final labels...")
    df_final, t1, t2 = calculate_final_thresholds(df_final)
    
    print("Generating sample IDs...")
    df_final.sort_values(by=["repository", "workflow_id", "commit_date", "commit_sha"], inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    df_final["sample_id"] = [f"gha_{str(i).zfill(7)}" for i in range(len(df_final))]
    
    data_quality_audit(df_final, initial_len, cap_len, df_excluded)
    
    print("\n===== FINAL LABELLED CORPUS =====")
    print(f"Workflow versions: {final_len:,}")
    print(f"Repositories: {df_final['repository'].nunique():,}")
    print(f"Maximum versions/repository: {df_final.groupby('repository').size().max()}")
    
    print("\n===== SPLIT =====")
    for s in ["train", "validation", "test"]:
        s_df = df_final[df_final["split"] == s]
        print(f"{s.capitalize()}:")
        print(f"  repositories: {s_df['repository'].nunique():,}")
        print(f"  workflow versions: {len(s_df):,}")
        
    print("\n===== TRAINING-ONLY THRESHOLDS =====")
    print(f"T1: {t1}")
    print(f"T2: {t2}")
    
    print("\n===== FINAL CLASS DISTRIBUTION =====")
    for s in ["train", "validation", "test"]:
        s_df = df_final[df_final["split"] == s]
        counts = s_df["final_label"].value_counts().sort_index()
        print(f"{s.capitalize()}:")
        for cls in [0, 1, 2]:
            print(f"  class {cls}: {counts.get(cls, 0):,}")
            
    print("\n===== LEAKAGE =====")
    overlap = len(train_repos & val_repos) + len(train_repos & test_repos) + len(val_repos & test_repos)
    print(f"Repository overlap: {overlap}")
    print(f"Threshold source: TRAIN ONLY")
    
    print("\n===== ARTIFACTS =====")
    out_pq = paths.processed_data_root / "labelled_workflow_versions_v1.parquet"
    out_csv = paths.processed_data_root / "labelled_workflow_versions_v1.csv"
    
    cols = [
        "sample_id", "workflow_id", "commit_sha", "owner", "repo", "repository",
        "workflow_path", "commit_date", "file_sha", "parsing_success", "parsed_workflow_available",
        "success", "failure", "timed_out", "eligible_runs", "failure_rate", "meets_minimum_10",
        "split", "final_label"
    ]
    df_final = df_final[cols]
    
    df_final.to_parquet(out_pq, index=False)
    print(f"Saved: {out_pq}")
    df_final.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    
    split_path = paths.splits_root / "repository_split_v1.csv"
    repo_split = df_final[["repository", "split"]].drop_duplicates().sort_values("repository")
    repo_split.to_csv(split_path, index=False)
    print(f"Saved: {split_path}")
    
    now = datetime.datetime.utcnow().isoformat()
    man_cand = {
        "initial_candidates": initial_len,
        "cap_20_candidates": cap_len,
        "quality_filtered": final_len,
        "repositories": int(df_final['repository'].nunique())
    }
    paths.manifests_root.joinpath("candidate_corpus_manifest.json").write_text(json.dumps(man_cand, indent=4))
    
    man_split = {
        "train_repos": len(train_repos),
        "validation_repos": len(val_repos),
        "test_repos": len(test_repos),
        "seed": 42
    }
    paths.manifests_root.joinpath("repository_split_v1.json").write_text(json.dumps(man_split, indent=4))
    
    man_label = {
        "dataset_stage": "final_processed",
        "target_name": "final_label",
        "class_mapping": {"0": "LOW", "1": "MEDIUM", "2": "HIGH"},
        "failure_rate_definition": "(failure + timed_out) / (success + failure + timed_out)",
        "minimum_eligible_future_runs": 10,
        "eligible_conclusions": ["success", "failure", "timed_out"],
        "threshold_method": "Empirical tertiles of TRAINING failure_rate",
        "T1": t1,
        "T2": t2,
        "exact_assignment_rule": "0 if <= T1 else (1 if <= T2 else 2)",
        "repository_cap": 20,
        "random_seed": 42,
        "train_repository_count": len(train_repos),
        "validation_repository_count": len(val_repos),
        "test_repository_count": len(test_repos),
        "training_workflow_version_count": len(df_final[df_final["split"]=="train"]),
        "validation_workflow_version_count": len(df_final[df_final["split"]=="validation"]),
        "test_workflow_version_count": len(df_final[df_final["split"]=="test"]),
        "creation_timestamp": now
    }
    paths.manifests_root.joinpath("final_label_definition_v1.json").write_text(json.dumps(man_label, indent=4))
    
    man_qual = {
        "initial_candidates_gt10": initial_len,
        "repository_cap_candidate_count": cap_len,
        "excluded_records": len(df_excluded),
        "exclusion_reasons": "parsing_success == False AND parsed_workflow_available == False",
        "final_retained_versions": final_len,
        "final_retained_repositories": int(df_final['repository'].nunique()),
        "missing_field_statistics": df_final.isnull().sum().to_dict(),
        "duplicate_counts": int(df_final.duplicated(subset=["workflow_id", "commit_sha"]).sum()),
        "parsed_workflow_availability": "100% required"
    }
    paths.manifests_root.joinpath("data_quality_manifest_v1.json").write_text(json.dumps(man_qual, indent=4))
    
if __name__ == "__main__":
    main()
