import sys
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import ProjectPaths
from src.features.structural_features import extract_workflow_features

def main():
    paths = ProjectPaths()
    input_pq = paths.processed_data_root / "labelled_workflow_versions_text_v1.parquet"
    out_pq = paths.processed_data_root / "structural_features_v1.parquet"
    
    print(f"Loading {input_pq}...")
    df = pd.read_parquet(input_pq)
    
    features_list = []
    
    print("Extracting features...")
    for idx, row in df.iterrows():
        pw = json.loads(row["parsed_workflow_json"])
        feat = extract_workflow_features(pw)
        feat["sample_id"] = row["sample_id"]
        feat["workflow_id"] = row["workflow_id"]
        feat["commit_sha"] = row["commit_sha"]
        features_list.append(feat)
        
    df_feat = pd.DataFrame(features_list)
    
    print("Performing Quality Audit...")
    
    # Identify feature columns
    feat_cols = [c for c in df_feat.columns if c not in ["sample_id", "workflow_id", "commit_sha"]]
    
    audit_results = []
    
    for c in feat_cols:
        col_data = df_feat[c]
        missing = col_data.isna().sum()
        unique = col_data.nunique()
        
        c_min = float(col_data.min()) if not pd.isna(col_data.min()) else 0.0
        c_max = float(col_data.max()) if not pd.isna(col_data.max()) else 0.0
        
        audit_results.append({
            "feature": c,
            "missing": int(missing),
            "unique_values": int(unique),
            "min": c_min,
            "max": c_max,
            "is_constant": unique <= 1,
            "is_near_constant": unique == 2 and col_data.value_counts(normalize=True).iloc[0] > 0.99
        })
        
    df_audit = pd.DataFrame(audit_results)
    
    print(f"\nTotal Features: {len(feat_cols)}")
    print(f"Constant features: {df_audit['is_constant'].sum()}")
    print(f"Near-constant features (>99% single value): {df_audit['is_near_constant'].sum()}")
    print(f"Features with missing values: {(df_audit['missing'] > 0).sum()}")
    
    print("\nLeakage Audit: Validating no leakage features exist...")
    forbidden = ["failure_rate", "success", "failure", "timed_out", "eligible_runs", "final_label", "provisional_class"]
    for f in forbidden:
        assert f not in df_feat.columns, f"LEAKAGE DETECTED: {f}"
        
    print(f"Saving to {out_pq}")
    df_feat.to_parquet(out_pq, index=False)
    
    # Manifest
    now = datetime.datetime.utcnow().isoformat()
    manifest = {
        "dataset_stage": "structural_features",
        "row_count": len(df_feat),
        "feature_count": len(feat_cols),
        "features": feat_cols,
        "audit_summary": {
            "constant_features": int(df_audit['is_constant'].sum()),
            "near_constant_features": int(df_audit['is_near_constant'].sum()),
            "missing_values_detected": int((df_audit['missing'] > 0).sum())
        },
        "creation_timestamp": now
    }
    paths.manifests_root.joinpath("structural_feature_manifest_v1.json").write_text(json.dumps(manifest, indent=4))
    
if __name__ == "__main__":
    main()
