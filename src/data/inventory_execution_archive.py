import sys
import zipfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import ProjectPaths

def inventory_zip():
    paths = ProjectPaths()
    zip_path = paths.raw_data_root / "github-workflows-december-2025.zip"
    
    if not zip_path.exists():
        print(f"Error: Archive not found at {zip_path}")
        sys.exit(1)
        
    print(f"Inspecting archive: {zip_path.name}")
    
    expected_collections = {
        "workflow_files.bson": False,
        "workflow_files_commits.bson": False,
        "workflow_runs.bson": False,
        "jobs_of_runs.bson": False
    }
    
    total_entries = 0
    total_compressed = 0
    total_uncompressed = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            infos = zf.infolist()
            total_entries = len(infos)
            print(f"Total entries: {total_entries}\n")
            print(f"{'Filename':<40} {'Compressed':<15} {'Uncompressed':<15} {'Extension':<10}")
            print("-" * 80)
            
            for info in infos:
                total_compressed += info.compress_size
                total_uncompressed += info.file_size
                filename = Path(info.filename)
                
                # Check for expected collections
                if filename.name in expected_collections:
                    expected_collections[filename.name] = True
                    
                print(f"{filename.name:<40} {info.compress_size:<15} {info.file_size:<15} {filename.suffix:<10}")
                
        print("-" * 80)
        print(f"{'TOTAL':<40} {total_compressed:<15} {total_uncompressed:<15}")
        
        print("\nVerification of expected collections:")
        all_present = True
        for coll, present in expected_collections.items():
            status = "FOUND" if present else "MISSING"
            print(f" - {coll}: {status}")
            if not present:
                all_present = False
                
        if all_present:
            print("\nSTATUS: PASS - All expected BSON collections are present.")
        else:
            print("\nSTATUS: FAIL - Missing expected BSON collections.")
            sys.exit(1)
            
    except zipfile.BadZipFile:
        print("Error: The file is not a valid zip archive.")
        sys.exit(1)

if __name__ == "__main__":
    inventory_zip()
