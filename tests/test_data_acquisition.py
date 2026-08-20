import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import ProjectPaths
from src.data.download_raw import acquire_raw_data, SOURCES

def test_path_configuration():
    """Test that path configuration initializes directories correctly."""
    paths = ProjectPaths()
    assert hasattr(paths, 'raw_data_root')
    assert hasattr(paths, 'manifests_root')
    # Make sure we don't accidentally write to root
    assert "data" in str(paths.raw_data_root)

@patch('src.data.download_raw.download_file')
@patch('src.data.download_raw.compute_sha256')
def test_manifest_creation_and_source_check(mock_sha, mock_download, tmp_path):
    """Test manifest creation and skip logic without actual downloading."""
    
    # Setup mock paths to point to tmp_path
    with patch('src.data.download_raw.ProjectPaths') as MockPaths:
        mock_paths_instance = MagicMock()
        mock_paths_instance.root = tmp_path
        mock_paths_instance.raw_data_root = tmp_path / "data" / "raw"
        mock_paths_instance.manifests_root = tmp_path / "data" / "manifests"
        mock_paths_instance.ensure_directories.side_effect = lambda: (
            mock_paths_instance.raw_data_root.mkdir(parents=True, exist_ok=True),
            mock_paths_instance.manifests_root.mkdir(parents=True, exist_ok=True)
        )
        MockPaths.return_value = mock_paths_instance
        
        mock_sha.return_value = "dummy_sha256"
        
        # We need to simulate the downloaded file existing after download_file is called
        def simulate_download(url, dest_path):
            with open(dest_path, "w") as f:
                f.write("dummy content")
        
        mock_download.side_effect = simulate_download
        
        # Run the acquisition logic
        acquire_raw_data()
        
        # 1. Check if files were "downloaded"
        assert mock_download.call_count == len(SOURCES)
        
        # 2. Check if manifest is created correctly
        manifest_file = mock_paths_instance.manifests_root / "raw_data_manifest.json"
        assert manifest_file.exists()
        
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
            
        assert len(manifest) == len(SOURCES)
        assert manifest[0]["status"] == "downloaded"
        assert manifest[0]["checksum_sha256"] == "dummy_sha256"
        
        # Run again to test existence check (skip download)
        mock_download.reset_mock()
        acquire_raw_data()
        
        # Should NOT have called download_file again
        assert mock_download.call_count == 0
        
        # Status should now be 'verified'
        with open(manifest_file, "r") as f:
            manifest2 = json.load(f)
        assert manifest2[0]["status"] == "verified"

@patch('src.data.inventory_execution_archive.ProjectPaths')
@patch('zipfile.ZipFile')
def test_zip_inventory_detection(mock_zip, MockPaths, tmp_path):
    """Test ZIP inventory logic with mock zipfile."""
    from src.data.inventory_execution_archive import inventory_zip
    
    # Mock paths
    mock_paths_instance = MagicMock()
    mock_paths_instance.raw_data_root = tmp_path
    MockPaths.return_value = mock_paths_instance
    
    # Create fake zip file to pass exists() check
    fake_zip = tmp_path / "github-workflows-december-2025.zip"
    fake_zip.touch()
    
    # Mock zipfile contents
    mock_zf = MagicMock()
    mock_info = MagicMock()
    mock_info.filename = "workflow_files.bson"
    mock_info.compress_size = 100
    mock_info.file_size = 200
    mock_zf.infolist.return_value = [mock_info]
    
    # Context manager setup
    mock_zip.return_value.__enter__.return_value = mock_zf
    
    # Should catch SystemExit since not all collections are present
    with pytest.raises(SystemExit) as excinfo:
        inventory_zip()
    assert excinfo.value.code == 1
    
    # Now provide all expected files
    infos = []
    for f in ["workflow_files.bson", "workflow_files_commits.bson", "workflow_runs.bson", "jobs_of_runs.bson"]:
        mi = MagicMock()
        mi.filename = f
        mi.compress_size = 100
        mi.file_size = 200
        infos.append(mi)
    mock_zf.infolist.return_value = infos
    
    # Should succeed without exiting
    try:
        inventory_zip()
    except SystemExit:
        pytest.fail("inventory_zip raised SystemExit unexpectedly!")

@patch('src.data.verify_secondary_csv.ProjectPaths')
@patch('pandas.read_csv')
def test_csv_sample_validation(mock_read_csv, MockPaths, tmp_path):
    """Test CSV validation logic with mock dataframe."""
    from src.data.verify_secondary_csv import verify_csv
    
    # Mock paths
    mock_paths_instance = MagicMock()
    mock_paths_instance.raw_data_root = tmp_path
    MockPaths.return_value = mock_paths_instance
    
    # Create fake csv
    fake_csv = tmp_path / "workflows.csv.gz"
    fake_csv.touch()
    
    # Mock pandas read_csv
    import pandas as pd
    mock_df = pd.DataFrame(columns=[
        "repository", "commit_hash", "author_name", "author_email",
        "committer_name", "committer_email", "committed_date", "authored_date",
        "file_path", "previous_file_path", "file_hash", "previous_file_hash",
        "git_change_type", "valid_yaml", "probably_workflow", "valid_workflow", "uid"
    ])
    mock_read_csv.return_value = mock_df
    
    # Should pass
    try:
        verify_csv()
    except SystemExit:
        pytest.fail("verify_csv raised SystemExit unexpectedly!")
        
    # Test failure mode
    mock_df_fail = pd.DataFrame(columns=["repository", "commit_hash"])
    mock_read_csv.return_value = mock_df_fail
    with pytest.raises(SystemExit):
        verify_csv()
