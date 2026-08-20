import sys
import zipfile
import json
import struct
import datetime
from pathlib import Path
from bisect import bisect_right
from collections import defaultdict
import pandas as pd
import bson

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import ProjectPaths

def robust_bson_stream(file_obj):
    """
    Generator that safely yields decoded BSON documents from a file-like object.
    Yields (doc, error) where error is None if successful, or Exception if failed.
    """
    while True:
        # Read the first 4 bytes to get document length
        length_bytes = file_obj.read(4)
        if not length_bytes or len(length_bytes) < 4:
            break
            
        try:
            doc_len = struct.unpack("<i", length_bytes)[0]
            if doc_len < 5 or doc_len > 16 * 1024 * 1024:
                # Sanity check: doc must be >= 5 bytes, and let's cap at 16MB standard limit
                raise ValueError(f"Invalid BSON document length: {doc_len}")
                
            remainder = file_obj.read(doc_len - 4)
            if len(remainder) != doc_len - 4:
                raise ValueError("Unexpected EOF while reading BSON document")
                
            doc_data = length_bytes + remainder
            # We explicitly decode with codec_options to handle datetimes
            doc = bson.BSON(doc_data).decode()
            yield doc, None
            
        except Exception as e:
            yield None, e

def load_workflow_commit_index(zf: zipfile.ZipFile, commit_filename: str):
    """
    Pass 1: Build in-memory index mapping workflow_id to a sorted list of (timestamp, commit_sha).
    """
    print(f"PASS 1: Loading {commit_filename}...")
    
    # Structure: dict[workflow_id] = list of (timestamp_int, commit_sha)
    commit_index = defaultdict(list)
    
    scanned = 0
    failures = 0
    
    with zf.open(commit_filename, "r") as f:
        for doc, err in robust_bson_stream(f):
            scanned += 1
            if err:
                failures += 1
                continue
                
            workflow_id = str(doc.get("workflow_id"))
            commit_sha = str(doc.get("commit_sha"))
            created_at = doc.get("commit_date")
            
            if workflow_id and commit_sha and created_at:
                # Normalize timestamp to float/int
                if isinstance(created_at, datetime.datetime):
                    # Handle naive vs aware datetimes
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                    ts = created_at.timestamp()
                elif isinstance(created_at, (int, float)):
                    ts = float(created_at)
                elif isinstance(created_at, str):
                    try:
                        # Simple parse if it's ISO
                        dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        ts = dt.timestamp()
                    except ValueError:
                        failures += 1
                        continue
                else:
                    failures += 1
                    continue
                    
                commit_index[workflow_id].append((ts, commit_sha))
                
            if scanned % 250000 == 0:
                print(f"  commits scanned: {scanned:,}")
                
    # Sort the lists by timestamp
    for w_id in commit_index:
        commit_index[w_id].sort(key=lambda x: x[0])
        
    print(f"PASS 1 Complete. Scanned: {scanned:,} | Failures: {failures:,} | Unique workflows: {len(commit_index):,}")
    return commit_index

def assign_run_to_workflow_version(commit_history: list, run_ts: float):
    """
    Uses binary search to find the latest workflow commit whose timestamp is <= run_ts.
    Returns commit_sha or None.
    """
    if not commit_history:
        return None
        
    # We only want the timestamps for bisect
    timestamps = [x[0] for x in commit_history]
    
    idx = bisect_right(timestamps, run_ts)
    if idx == 0:
        # Run happened before the FIRST known commit for this workflow
        return None
        
    # Return the commit_sha of the preceding commit
    return commit_history[idx - 1][1]

def process_workflow_runs(zf: zipfile.ZipFile, runs_filename: str, commit_index: dict):
    """
    Pass 2: Stream runs, assign to active workflow version, and aggregate.
    """
    print(f"\nPASS 2: Processing {runs_filename}...")
    
    # Structure: dict[(workflow_id, commit_sha)] = {"success": 0, "failure": 0, "timed_out": 0}
    aggregates = defaultdict(lambda: {"success": 0, "failure": 0, "timed_out": 0})
    
    scanned = 0
    failures = 0
    eligible_outcomes = 0
    assigned = 0
    
    valid_conclusions = {"success", "failure", "timed_out"}
    
    with zf.open(runs_filename, "r") as f:
        for doc, err in robust_bson_stream(f):
            scanned += 1
            if err:
                failures += 1
                continue
                
            conclusion = doc.get("conclusion")
            if conclusion not in valid_conclusions:
                continue
                
            eligible_outcomes += 1
            
            workflow_id = str(doc.get("workflow_id"))
            created_at = doc.get("created_at")
            
            if workflow_id not in commit_index or not created_at:
                continue
                
            # Normalize run timestamp
            if isinstance(created_at, datetime.datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                ts = created_at.timestamp()
            elif isinstance(created_at, (int, float)):
                ts = float(created_at)
            elif isinstance(created_at, str):
                try:
                    dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    ts = dt.timestamp()
                except ValueError:
                    failures += 1
                    continue
            else:
                failures += 1
                continue
                
            # Assign
            active_commit_sha = assign_run_to_workflow_version(commit_index[workflow_id], ts)
            if active_commit_sha:
                aggregates[(workflow_id, active_commit_sha)][conclusion] += 1
                assigned += 1
                
            if scanned % 250000 == 0:
                print(f"  workflow runs scanned: {scanned:,} | eligible: {eligible_outcomes:,} | assigned: {assigned:,}")
                
    print(f"PASS 2 Complete.")
    print(f"  Total Scanned: {scanned:,}")
    print(f"  Decode Failures: {failures:,}")
    print(f"  Eligible Outcomes: {eligible_outcomes:,}")
    print(f"  Assigned Runs: {assigned:,}")
    
    return aggregates, scanned, failures, eligible_outcomes, assigned

def build_execution_label_table(aggregates: dict):
    """
    Converts aggregated counts into a validation-ready pandas DataFrame.
    """
    rows = []
    for (w_id, c_sha), counts in aggregates.items():
        succ = counts["success"]
        fail = counts["failure"]
        tout = counts["timed_out"]
        elig = succ + fail + tout
        
        # Guard against zero eligible runs (should be impossible based on logic, but be safe)
        if elig == 0:
            continue
            
        fail_rate = (fail + tout) / elig
        meets_10 = elig >= 10
        
        rows.append({
            "workflow_id": w_id,
            "commit_sha": c_sha,
            "success": succ,
            "failure": fail,
            "timed_out": tout,
            "eligible_runs": elig,
            "failure_rate": fail_rate,
            "meets_minimum_10": meets_10
        })
        
    df = pd.DataFrame(rows)
    return df

def validate_dataframe(df: pd.DataFrame):
    """
    Run the requested assertions.
    """
    print("\nValidating Dataframe...")
    assert not df.duplicated(subset=["workflow_id", "commit_sha"]).any(), "Duplicate keys found!"
    assert (df["success"] >= 0).all() and (df["failure"] >= 0).all() and (df["timed_out"] >= 0).all(), "Negative counts found!"
    assert (df["eligible_runs"] == df["success"] + df["failure"] + df["timed_out"]).all(), "Inconsistent eligible_runs!"
    assert df["failure_rate"].between(0, 1).all(), "Failure rate out of bounds!"
    assert (df["meets_minimum_10"] == (df["eligible_runs"] >= 10)).all(), "meets_minimum_10 incorrect!"
    assert df["workflow_id"].notna().all(), "Missing workflow_id!"
    assert df["commit_sha"].notna().all(), "Missing commit_sha!"
    print("Validation passed.")

def main():
    paths = ProjectPaths()
    zip_path = paths.raw_data_root / "github-workflows-december-2025.zip"
    
    if not zip_path.exists():
        print(f"Raw data zip not found: {zip_path}")
        sys.exit(1)
        
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Expected collections inside ZIP
        commit_bson = "github-workflows-december-2025/workflow_files_commits.bson"
        runs_bson = "github-workflows-december-2025/workflow_runs.bson"
        
        commit_index = load_workflow_commit_index(zf, commit_bson)
        
        # Stop early if catastrophic failure
        if not commit_index:
            print("No commits parsed. Aborting.")
            sys.exit(1)
            
        aggregates, r_scanned, r_fail, r_elig, r_assign = process_workflow_runs(zf, runs_bson, commit_index)
        
    df = build_execution_label_table(aggregates)
    validate_dataframe(df)
    
    # Save artifacts
    out_pq = paths.intermediate_data_root / "execution_label_coverage_v1.parquet"
    out_csv = paths.intermediate_data_root / "execution_label_coverage_v1.csv"
    
    df.to_parquet(out_pq, index=False)
    print(f"Saved to {out_pq}")
    
    # Only save CSV if reasonably sized. Here df length should be ~42k
    if len(df) < 500_000:
        df.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")
        
    # Build Manifest
    versions_any = len(df)
    versions_10 = df["meets_minimum_10"].sum()
    
    manifest = {
        "source_zip_path": str(zip_path.relative_to(paths.root)),
        "commit_collection_name": commit_bson,
        "run_collection_name": runs_bson,
        "eligible_outcomes": ["success", "failure", "timed_out"],
        "minimum_eligible_runs": 10,
        "temporal_assignment_rule": "latest commit_timestamp <= run_created_at",
        "number_of_commits_scanned": sum(len(v) for v in commit_index.values()),
        "number_of_runs_scanned": r_scanned,
        "number_of_eligible_runs": r_elig,
        "number_of_assigned_runs": r_assign,
        "number_of_decode_failures": r_fail,
        "number_of_workflow_versions": versions_any,
        "number_of_usable_gt10_versions": int(versions_10),
        "build_timestamp": datetime.datetime.utcnow().isoformat(),
        "random_seed": None,
        "code_version": "v1.0"
    }
    
    man_path = paths.manifests_root / "execution_label_build_manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\nFINAL STATISTICS:")
    print(f"  workflow versions with >= 1 eligible execution: {versions_any:,}")
    print(f"  workflow versions with >= 10 eligible executions: {versions_10:,}")

if __name__ == "__main__":
    main()
