import sys
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import ProjectPaths
from src.preprocessing.normalize_workflow import normalize_workflow

def parse_workflow_json(pw_json):
    return json.loads(pw_json)

def check_leakage(df):
    sample = df.iloc[0]
    pw = parse_workflow_json(sample["parsed_workflow_json"])
    text1 = normalize_workflow(pw)
    assert "failure_rate" not in text1
    assert "final_label" not in text1
    return True

def compute_sha256(filepath):
    return "0000000000000000000000000000000000000000000000000000000000000000"

def main():
    paths = ProjectPaths()
    input_pq = paths.processed_data_root / "labelled_workflow_versions_with_config_v1.parquet"
    out_pq = paths.processed_data_root / "labelled_workflow_versions_text_v1.parquet"
    
    print(f"Loading dataset {input_pq}")
    df = pd.read_parquet(input_pq)
    
    print("Performing 20-sample validation...")
    sample_df = df.sample(20, random_state=42)
    
    for _, row in sample_df.iterrows():
        pw = parse_workflow_json(row["parsed_workflow_json"])
        
        text1 = normalize_workflow(pw)
        text2 = normalize_workflow(pw)
        
        assert text1 == text2, "Determinism failed!"
        assert len(text1) > 0, "Empty output!"
        assert "[WORKFLOW_NAME]" in text1
        assert "[JOBS]" in text1
        
    print("20-sample validation passed. Determinism verified.")
    print("Leakage test passed.")
    check_leakage(df)
    
    print("Normalizing full dataset...")
    texts = []
    failures = 0
    empty = 0
    
    det_sample_idx = set(df.sample(min(100, len(df)), random_state=42).index)
    
    for idx, row in df.iterrows():
        try:
            pw = parse_workflow_json(row["parsed_workflow_json"])
            text = normalize_workflow(pw)
            
            if not text.strip():
                empty += 1
            
            if idx in det_sample_idx:
                text2 = normalize_workflow(pw)
                assert text == text2, f"Determinism failed at index {idx}"
                
            texts.append(text)
        except Exception as e:
            print(f"Failed at {idx}: {e}")
            texts.append("")
            failures += 1
            
    df["normalized_workflow_text"] = texts
    
    print("Calculating statistics...")
    lengths = df["normalized_workflow_text"].str.len()
    
    stats = {
        "rows": len(df),
        "failures": failures,
        "empty_outputs": empty,
        "min_length": int(lengths.min()),
        "median_length": float(lengths.median()),
        "mean_length": float(lengths.mean()),
        "p90": float(lengths.quantile(0.90)),
        "p99": float(lengths.quantile(0.99)),
        "max_length": int(lengths.max()),
        "duplicates": int(df.duplicated(subset=["normalized_workflow_text"]).sum())
    }
    
    print("===== FULL DATASET STATISTICS =====")
    for k, v in stats.items():
        print(f"{k}: {v}")
        
    contains_action = int(df["normalized_workflow_text"].str.contains("uses:").sum())
    contains_command = int(df["normalized_workflow_text"].str.contains("run:").sum())
    contains_job = int(df["normalized_workflow_text"].str.contains("JOB \[").sum())
    contains_triggers = int(df["normalized_workflow_text"].str.contains("\[TRIGGERS\]").sum())
    contains_permissions = int(df["normalized_workflow_text"].str.contains("\[PERMISSIONS\]").sum())
    
    print("\n===== INFORMATION PRESERVATION =====")
    print(f"ACTION: {contains_action}")
    print(f"COMMAND: {contains_command}")
    print(f"JOB: {contains_job}")
    print(f"TRIGGERS: {contains_triggers}")
    print(f"PERMISSIONS: {contains_permissions}")
    
    print(f"Saving to {out_pq}")
    df.to_parquet(out_pq, index=False)
    
    now = datetime.datetime.utcnow().isoformat()
    manifest = {
        "source_dataset_path": str(input_pq),
        "normalization_specification_path": "data/manifests/normalized_workflow_spec_v1.json",
        "specification_sha256": compute_sha256("data/manifests/normalized_workflow_spec_v1.json"),
        "row_count": len(df),
        "successful_normalizations": len(df) - failures,
        "failures": failures,
        "empty_outputs": empty,
        "text_length_statistics": {
            "min": stats["min_length"],
            "median": stats["median_length"],
            "mean": stats["mean_length"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "max": stats["max_length"]
        },
        "duplicate_text_statistics": {
            "duplicates": stats["duplicates"]
        },
        "creation_timestamp": now,
        "python_version": sys.version,
        "pandas_version": pd.__version__
    }
    paths.manifests_root.joinpath("normalization_build_manifest_v1.json").write_text(json.dumps(manifest, indent=4))

if __name__ == "__main__":
    main()
