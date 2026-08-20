import pytest
import pandas as pd
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# We will test functions from src.data.extract_workflow_configs

def test_json_serialization_round_trip():
    import json
    dummy_dict = {"name": "test", "jobs": {"build": {"runs-on": "ubuntu-latest"}}}
    
    # serialization
    json_str = json.dumps(dummy_dict)
    
    # round-trip
    recovered_dict = json.loads(json_str)
    assert recovered_dict == dummy_dict

def test_missing_parsed_workflow_rejection():
    # If a parsed_workflow is missing or empty, it should raise or be caught
    from src.data.extract_workflow_configs import validate_parsed_workflow
    
    # Valid
    assert validate_parsed_workflow({"jobs": {}}) == True
    
    # Invalid
    assert validate_parsed_workflow(None) == False
    assert validate_parsed_workflow({}) == False
    assert validate_parsed_workflow("string") == False

def test_merge_preserves_row_count_and_labels():
    from src.data.extract_workflow_configs import merge_config_to_labels
    
    df_labels = pd.DataFrame({
        "workflow_id": ["1", "2"],
        "commit_sha": ["a", "b"],
        "final_label": [0, 1],
        "split": ["train", "validation"], "failure_rate": [0.1, 0.2]
    })
    
    df_configs = pd.DataFrame({
        "workflow_id": ["1", "2"],
        "commit_sha": ["a", "b"],
        "parsed_workflow_json": ['{"test":1}', '{"test":2}']
    })
    
    df_merged = merge_config_to_labels(df_labels, df_configs)
    
    assert len(df_merged) == len(df_labels)
    assert list(df_merged["final_label"]) == [0, 1]
    assert list(df_merged["split"]) == ["train", "validation"]
    assert "parsed_workflow_json" in df_merged.columns

def test_detect_duplicate_conflicts():
    from src.data.extract_workflow_configs import check_for_duplicates
    
    df_configs = pd.DataFrame({
        "workflow_id": ["1", "1"],
        "commit_sha": ["a", "a"],
        "parsed_workflow_json": ['{"test":1}', '{"test":1}']
    })
    
    # Should not raise if exact duplicate
    check_for_duplicates(df_configs)
    
    df_configs_conflict = pd.DataFrame({
        "workflow_id": ["1", "1"],
        "commit_sha": ["a", "a"],
        "parsed_workflow_json": ['{"test":1}', '{"test":2}']
    })
    
    # Should raise on conflicting content
    with pytest.raises(ValueError, match="conflicting content"):
        check_for_duplicates(df_configs_conflict)

