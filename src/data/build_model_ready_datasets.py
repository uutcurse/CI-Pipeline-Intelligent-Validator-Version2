import sys
import json
import datetime
import pandas as pd
from pathlib import Path

def main():
    root = Path("data/processed")
    text_pq = root / "labelled_workflow_versions_text_v1.parquet"
    struct_pq = root / "structural_features_v1.parquet"
    
    print("Loading data...")
    df_text = pd.read_parquet(text_pq)
    df_struct = pd.read_parquet(struct_pq)
    
    # 1. Validate Join Keys
    assert len(df_text) == 12944, f"Text length mismatch: {len(df_text)}"
    assert len(df_struct) == 12944, f"Structure length mismatch: {len(df_struct)}"
    
    assert df_text["sample_id"].is_unique, "Duplicate sample_ids in text"
    assert df_struct["sample_id"].is_unique, "Duplicate sample_ids in structure"
    
    # Inner join on sample_id
    # We must not duplicate structural columns or provenance columns
    # df_struct already contains workflow_id and commit_sha, let's drop them before join to avoid _x / _y
    df_struct_clean = df_struct.drop(columns=["workflow_id", "commit_sha"])
    
    df_joined = pd.merge(df_text, df_struct_clean, on="sample_id", how="inner")
    
    assert len(df_joined) == 12944, f"Join size mismatch: {len(df_joined)}"
    
    # Define Column Sets
    forbidden_features = [
        "success", "failure", "timed_out", "eligible_runs", "failure_rate",
        "meets_minimum_10", "observation_days", "provisional_class"
    ]
    
    provenance_cols = [
        "sample_id", "repository", "owner", "repo", "workflow_id",
        "commit_sha", "commit_date", "workflow_path", "file_sha", "split", "final_label"
    ]
    
    audit_cols = [
        "parsing_success", "parsed_workflow_available", "parsed_workflow_json"
    ] + [f for f in forbidden_features if f in df_joined.columns]
    
    text_cols = ["sample_id", "normalized_workflow_text", "final_label", "split"]
    
    struct_feat_cols = [c for c in df_struct_clean.columns if c != "sample_id"]
    
    struct_view_cols = ["sample_id"] + struct_feat_cols + ["final_label", "split"]
    
    hybrid_view_cols = ["sample_id", "normalized_workflow_text"] + struct_feat_cols + ["final_label", "split"]
    
    prov_view_cols = provenance_cols + audit_cols
    
    # Assert no forbidden features in model views
    for f in forbidden_features:
        assert f not in text_cols
        assert f not in struct_view_cols
        assert f not in hybrid_view_cols
        
    # Assert repository info not in model features
    assert "repository" not in struct_feat_cols
    assert "workflow_id" not in struct_feat_cols
    assert "commit_sha" not in struct_feat_cols
    
    print("Creating views...")
    view_text = df_joined[text_cols].copy()
    view_struct = df_joined[struct_view_cols].copy()
    view_hybrid = df_joined[hybrid_view_cols].copy()
    view_prov = df_joined[prov_view_cols].copy()
    
    print("Validating datasets...")
    assert len(view_text) == 12944
    assert not view_text["normalized_workflow_text"].isna().any()
    
    # Split checks
    for s in ["train", "validation", "test"]:
        sub = view_hybrid[view_hybrid["split"] == s]
        classes = sub["final_label"].unique()
        assert len(classes) == 3, f"Split {s} missing classes: {classes}"
        
    # Check repo leakage explicitly via provenance
    train_repos = set(view_prov[view_prov["split"] == "train"]["repository"])
    val_repos = set(view_prov[view_prov["split"] == "validation"]["repository"])
    test_repos = set(view_prov[view_prov["split"] == "test"]["repository"])
    assert len(train_repos & val_repos) == 0, "Train/Val repo leakage"
    assert len(train_repos & test_repos) == 0, "Train/Test repo leakage"
    assert len(val_repos & test_repos) == 0, "Val/Test repo leakage"
    
    # Statistics
    print("\n===== CORPUS STATISTICS =====")
    for s in ["train", "validation", "test"]:
        sub = view_prov[view_prov["split"] == s]
        print(f"Split: {s}")
        print(f"  Samples: {len(sub)}")
        print(f"  Repositories: {sub['repository'].nunique()}")
        counts = sub['final_label'].value_counts()
        print(f"  Class 0: {counts.get(0, 0)}")
        print(f"  Class 1: {counts.get(1, 0)}")
        print(f"  Class 2: {counts.get(2, 0)}")
        
    print("\nText lengths:")
    lengths = view_text["normalized_workflow_text"].str.len()
    print(f"  p50: {lengths.median()}")
    print(f"  p90: {lengths.quantile(0.90)}")
    print(f"  p99: {lengths.quantile(0.99)}")
    print(f"  max: {lengths.max()}")
    
    # Output Parquets
    print("Saving datasets...")
    view_text.to_parquet(root / "model_ready_text_v1.parquet", index=False)
    view_struct.to_parquet(root / "model_ready_structure_v1.parquet", index=False)
    view_hybrid.to_parquet(root / "model_ready_hybrid_v1.parquet", index=False)
    view_prov.to_parquet(root / "model_provenance_v1.parquet", index=False)
    
    # Build Contract Manifest
    man_root = Path("data/manifests")
    
    contract = {
        "dataset_stage": "model_feature_contract",
        "TEXT_FEATURES": ["normalized_workflow_text"],
        "STRUCTURAL_FEATURES": struct_feat_cols,
        "PROVENANCE_ONLY": provenance_cols,
        "LABEL_ONLY": ["final_label"],
        "FORBIDDEN_MODEL_FEATURES": forbidden_features,
        "feature_types": {}
    }
    
    for c in struct_feat_cols:
        col = view_struct[c]
        u = col.nunique()
        if u <= 2:
            ftype = "binary"
        else:
            ftype = "integer" if (col % 1 == 0).all() else "continuous"
        
        contract["feature_types"][c] = ftype
        
    man_root.joinpath("model_feature_contract_v1.json").write_text(json.dumps(contract, indent=4))
    
    # Build Experiment Manifest
    manifest = {
        "dataset_stage": "model_ready_datasets",
        "row_count": len(df_joined),
        "split_sizes": {
            "train": len(view_hybrid[view_hybrid["split"] == "train"]),
            "validation": len(view_hybrid[view_hybrid["split"] == "validation"]),
            "test": len(view_hybrid[view_hybrid["split"] == "test"])
        },
        "feature_count": len(struct_feat_cols),
        "text_representation_version": "v1",
        "source_dataset_versions": {
            "text": "data/processed/labelled_workflow_versions_text_v1.parquet",
            "structure": "data/processed/structural_features_v1.parquet",
            "split_manifest": "data/manifests/repository_split_v1.json",
            "final_label_definition": "data/manifests/final_label_definition_v1.json"
        },
        "exact_normalized_text_duplicate_count": 459, # Copied from Phase 2.3 audit
        "creation_timestamp": datetime.datetime.utcnow().isoformat(),
        "software_versions": {
            "python": sys.version,
            "pandas": pd.__version__
        }
    }
    
    man_root.joinpath("model_ready_dataset_manifest_v1.json").write_text(json.dumps(manifest, indent=4))
    
    print("Done!")

if __name__ == "__main__":
    main()
