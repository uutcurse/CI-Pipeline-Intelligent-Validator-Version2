import sys
import zipfile
import json
import datetime
import hashlib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import ProjectPaths
from src.data.build_execution_labels import robust_bson_stream

def validate_parsed_workflow(pw):
    """Returns True if the parsed workflow is a non-empty dictionary."""
    if not isinstance(pw, dict):
        return False
    if len(pw) == 0:
        return False
    return True

def merge_config_to_labels(df_labels, df_configs):
    """Merge keeping labels intact."""
    initial_len = len(df_labels)
    
    # We only need to merge the new config column, but let's allow overlapping columns like owner/repo
    # which will be suffixed if we don't drop them.
    # To keep it clean, let's only keep the new unique columns from configs
    # (plus keys for merge)
    config_cols = set(df_configs.columns)
    label_cols = set(df_labels.columns)
    
    # Drop columns from df_configs that already exist in df_labels, except the keys
    drop_cols = (config_cols & label_cols) - {"workflow_id", "commit_sha"}
    df_configs_clean = df_configs.drop(columns=list(drop_cols))
    
    df_merged = pd.merge(df_labels, df_configs_clean, on=["workflow_id", "commit_sha"], how="left")
    
    assert len(df_merged) == initial_len, "Merge changed row count!"
    assert df_merged["parsed_workflow_json"].notna().all(), "Missing parsed_workflow_json for some labels!"
    
    # Check that labels are unchanged
    assert df_merged["final_label"].equals(df_labels["final_label"])
    assert df_merged["split"].equals(df_labels["split"])
    assert df_merged["failure_rate"].equals(df_labels["failure_rate"])
    
    return df_merged

def check_for_duplicates(df_configs):
    """Check for duplicates and conflicts."""
    dupes = df_configs[df_configs.duplicated(subset=["workflow_id", "commit_sha"], keep=False)]
    if len(dupes) > 0:
        # Group by keys and check if parsed_workflow_json is identical
        for (w_id, c_sha), group in dupes.groupby(["workflow_id", "commit_sha"]):
            if group["parsed_workflow_json"].nunique() > 1:
                raise ValueError(f"Duplicate target with conflicting content: {w_id} {c_sha}")
    
    return df_configs.drop_duplicates(subset=["workflow_id", "commit_sha"])

def compute_sha256(filepath):
    # Avoid reading the entire 700MB into memory
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    paths = ProjectPaths()
    zip_path = paths.raw_data_root / "github-workflows-december-2025.zip"
    label_path = paths.processed_data_root / "labelled_workflow_versions_v1.parquet"
    
    print("Loading frozen labelled corpus...")
    df_labels = pd.read_parquet(label_path)
    df_labels["workflow_id"] = df_labels["workflow_id"].astype(str)
    df_labels["commit_sha"] = df_labels["commit_sha"].astype(str)
    
    expected_rows = len(df_labels)
    target_keys = set(zip(df_labels["workflow_id"], df_labels["commit_sha"]))
    print(f"Target versions to extract: {expected_rows}")
    
    scanned = 0
    failures = 0
    matches = 0
    duplicates = 0
    
    extracted = []
    
    print("Streaming BSON...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open("github-workflows-december-2025/workflow_files_commits.bson", "r") as f:
            for doc, err in robust_bson_stream(f):
                scanned += 1
                if err or not doc:
                    failures += 1
                    continue
                
                w_id = str(doc.get("workflow_id"))
                c_sha = str(doc.get("commit_sha"))
                
                if (w_id, c_sha) in target_keys:
                    pw = doc.get("parsed_workflow")
                    if not validate_parsed_workflow(pw):
                        print(f"ERROR: Missing/invalid parsed_workflow for target {w_id} {c_sha}")
                        sys.exit(1)
                    
                    pw_json = json.dumps(pw, default=str)
                    
                    record = {
                        "workflow_id": w_id,
                        "commit_sha": c_sha,
                        "owner": doc.get("owner"),
                        "repo": doc.get("repo"),
                        "workflow_path": doc.get("workflow_path"),
                        "commit_date": doc.get("commit_date"),
                        "file_sha": doc.get("file_sha"),
                        "file_size": doc.get("file_size"),
                        "parsing_success": bool(doc.get("parsing_success", False)),
                        "parsed_workflow_json": pw_json
                    }
                    extracted.append(record)
                    matches += 1
                
                if scanned % 250000 == 0:
                    print(f"  records scanned: {scanned:,} | matches: {matches:,}")
                    
    print(f"Streaming complete. Scanned: {scanned:,} | Matches: {matches:,}")
    
    df_configs = pd.DataFrame(extracted)
    df_configs["workflow_id"] = df_configs["workflow_id"].astype(str)
    df_configs["commit_sha"] = df_configs["commit_sha"].astype(str)
    
    # Check duplicates
    initial_match_count = len(df_configs)
    df_configs = check_for_duplicates(df_configs)
    dedup_count = len(df_configs)
    duplicates = initial_match_count - dedup_count
    
    missing = expected_rows - dedup_count
    print(f"Unique extracted configurations: {dedup_count}")
    print(f"Missing targets: {missing}")
    
    if missing > 0:
        print("ERROR: Missing target configurations!")
        sys.exit(1)
        
    print("Merging with labelled dataset...")
    df_merged = merge_config_to_labels(df_labels, df_configs)
    
    out_pq_inter = paths.intermediate_data_root / "workflow_configurations_v1.parquet"
    out_pq_final = paths.processed_data_root / "labelled_workflow_versions_with_config_v1.parquet"
    
    print("Saving intermediate configuration dataset...")
    df_configs.to_parquet(out_pq_inter, index=False)
    
    print("Saving merged dataset...")
    df_merged.to_parquet(out_pq_final, index=False)
    
    print("Hashing source zip...")
    zip_sha = compute_sha256(zip_path)
    
    now = datetime.datetime.utcnow().isoformat()
    manifest = {
        "source_zip_path": str(zip_path),
        "source_zip_sha256": zip_sha,
        "bson_collection": "github-workflows-december-2025/workflow_files_commits.bson",
        "labelled_corpus_path": str(label_path),
        "labelled_corpus_row_count": expected_rows,
        "records_scanned": scanned,
        "target_matches": matches,
        "duplicate_matches": duplicates,
        "decode_failures": failures,
        "missing_targets": missing,
        "parsed_workflow_coverage": "100%",
        "output_path": str(out_pq_final),
        "creation_timestamp": now,
        "python_version": sys.version,
        "pandas_version": pd.__version__
    }
    paths.manifests_root.joinpath("workflow_config_extraction_manifest_v1.json").write_text(json.dumps(manifest, indent=4))
    print("Manifest saved.")
    
if __name__ == "__main__":
    main()
