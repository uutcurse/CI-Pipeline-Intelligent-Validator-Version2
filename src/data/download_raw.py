import os
import sys
import json
import hashlib
import urllib.request
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import ProjectPaths

SOURCES = [
    {
        "logical_name": "execution_history",
        "url": "https://zenodo.org/records/18163515/files/github-workflows-december-2025.zip",
        "filename": "github-workflows-december-2025.zip",
        "expected_size_approx_mb": 676.83
    },
    {
        "logical_name": "workflow_metadata",
        "url": "https://zenodo.org/records/20340547/files/workflows.csv.gz",
        "filename": "workflows.csv.gz",
        "expected_size_approx_mb": 296.9
    }
]

def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url: str, dest_path: Path) -> None:
    """Downloads a file with streaming and basic progress reporting."""
    print(f"Downloading {url} to {dest_path}")
    
    # We use a progress hook for urllib
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end="")
        else:
            print(f"\rDownloaded {count * block_size} bytes", end="")
            
    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=progress_hook)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nFailed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def acquire_raw_data():
    paths = ProjectPaths()
    paths.ensure_directories()
    
    manifest_path = paths.manifests_root / "raw_data_manifest.json"
    manifest = []
    
    for source in SOURCES:
        dest_path = paths.raw_data_root / source["filename"]
        
        record = {
            "logical_name": source["logical_name"],
            "url": source["url"],
            "local_path": str(dest_path.relative_to(paths.root)),
            "expected_size_approx_mb": source["expected_size_approx_mb"],
            "status": "pending",
            "download_timestamp": None,
            "actual_size_bytes": None,
            "checksum_sha256": None,
            "file_extension": dest_path.suffix
        }
        
        # Check if file exists and has reasonable size
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"File {dest_path.name} already exists. Verifying...")
            actual_size = dest_path.stat().st_size
            record["actual_size_bytes"] = actual_size
            record["checksum_sha256"] = compute_sha256(dest_path)
            record["status"] = "verified"
            record["download_timestamp"] = datetime.utcnow().isoformat()
            print(f"Verified {dest_path.name} (Size: {actual_size} bytes)")
        else:
            download_file(source["url"], dest_path)
            actual_size = dest_path.stat().st_size
            record["actual_size_bytes"] = actual_size
            print("Computing SHA-256...")
            record["checksum_sha256"] = compute_sha256(dest_path)
            record["status"] = "downloaded"
            record["download_timestamp"] = datetime.utcnow().isoformat()
            print(f"Saved and verified {dest_path.name} (Size: {actual_size} bytes)")
            
        manifest.append(record)
        
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    acquire_raw_data()
