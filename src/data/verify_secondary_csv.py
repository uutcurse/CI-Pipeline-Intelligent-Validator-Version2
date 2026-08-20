import sys
import gzip
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import ProjectPaths

def verify_csv():
    paths = ProjectPaths()
    csv_path = paths.raw_data_root / "workflows.csv.gz"
    
    if not csv_path.exists():
        print(f"Error: Secondary dataset not found at {csv_path}")
        sys.exit(1)
        
    print(f"Inspecting secondary dataset: {csv_path.name}")
    
    try:
        # Report approximate compressed size
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"Compressed size: {size_mb:.2f} MB\n")
        
        # Read only a small sample without loading entire 4.2M-row CSV
        print("Reading sample (first 5 rows)...")
        df_sample = pd.read_csv(csv_path, compression='gzip', nrows=5)
        
        print("\nColumns:")
        for i, col in enumerate(df_sample.columns):
            print(f" {i+1}. {col}")
            
        print(f"\nSample Shape: {df_sample.shape}")
        
        # Confirm expected columns
        expected_columns = {
            "repository", "commit_hash", "author_name", "author_email",
            "committer_name", "committer_email", "committed_date", "authored_date",
            "file_path", "previous_file_path", "file_hash", "previous_file_hash",
            "git_change_type", "valid_yaml", "probably_workflow", "valid_workflow", "uid"
        }
        
        actual_columns = set(df_sample.columns)
        missing_columns = expected_columns - actual_columns
        
        if missing_columns:
            print(f"\nSTATUS: FAIL - Missing expected columns: {missing_columns}")
            sys.exit(1)
        else:
            print("\nSTATUS: PASS - All expected columns are present.")
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_csv()
