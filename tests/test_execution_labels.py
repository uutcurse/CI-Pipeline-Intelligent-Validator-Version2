import pytest
import pandas as pd
from src.data.build_execution_labels import (
    assign_run_to_workflow_version,
    build_execution_label_table,
    validate_dataframe
)

def test_assign_run_to_workflow_version():
    """
    Test binary search assignment logic.
    A run occurring after commit B must not be assigned to commit A if commit B is already active.
    """
    # commit_history is a sorted list of (timestamp, commit_sha)
    history = [
        (100.0, "commit_1"),
        (200.0, "commit_2"),
        (300.0, "commit_3")
    ]
    
    # Run before first commit -> None
    assert assign_run_to_workflow_version(history, 50.0) is None
    
    # Run exactly at first commit -> commit_1
    assert assign_run_to_workflow_version(history, 100.0) == "commit_1"
    
    # Run between commit 1 and 2 -> commit_1
    assert assign_run_to_workflow_version(history, 150.0) == "commit_1"
    
    # Run exactly at commit 2 -> commit_2
    assert assign_run_to_workflow_version(history, 200.0) == "commit_2"
    
    # Run between commit 2 and 3 -> commit_2
    assert assign_run_to_workflow_version(history, 299.9) == "commit_2"
    
    # Run after latest commit -> commit_3
    assert assign_run_to_workflow_version(history, 500.0) == "commit_3"
    
    # Empty history
    assert assign_run_to_workflow_version([], 500.0) is None

def test_build_execution_label_table():
    """Test aggregation correctness."""
    aggregates = {
        # Success only
        ("w1", "c1"): {"success": 5, "failure": 0, "timed_out": 0},
        # Failure only
        ("w2", "c2"): {"success": 0, "failure": 4, "timed_out": 0},
        # Mixed
        ("w3", "c3"): {"success": 5, "failure": 3, "timed_out": 2},
        # Exactly 10
        ("w4", "c4"): {"success": 8, "failure": 1, "timed_out": 1},
        # 10+ runs
        ("w5", "c5"): {"success": 50, "failure": 10, "timed_out": 5},
    }
    
    df = build_execution_label_table(aggregates)
    
    assert len(df) == 5
    
    w1 = df[df["workflow_id"] == "w1"].iloc[0]
    assert w1["eligible_runs"] == 5
    assert w1["failure_rate"] == 0.0
    assert w1["meets_minimum_10"] == False
    
    w2 = df[df["workflow_id"] == "w2"].iloc[0]
    assert w2["eligible_runs"] == 4
    assert w2["failure_rate"] == 1.0
    
    w3 = df[df["workflow_id"] == "w3"].iloc[0]
    assert w3["eligible_runs"] == 10
    assert w3["failure_rate"] == 0.5
    assert w3["meets_minimum_10"] == True
    
    w5 = df[df["workflow_id"] == "w5"].iloc[0]
    assert w5["eligible_runs"] == 65
    assert abs(w5["failure_rate"] - (15 / 65)) < 1e-5
    
    # Ensure validation passes
    validate_dataframe(df)

def test_validation_catches_errors():
    # Negative count
    bad_df = pd.DataFrame([{
        "workflow_id": "w1", "commit_sha": "c1", 
        "success": -1, "failure": 0, "timed_out": 0,
        "eligible_runs": -1, "failure_rate": 0, "meets_minimum_10": False
    }])
    with pytest.raises(AssertionError):
        validate_dataframe(bad_df)
        
    # Duplicate keys
    bad_df2 = pd.DataFrame([
        {"workflow_id": "w1", "commit_sha": "c1", "success": 1, "failure": 0, "timed_out": 0, "eligible_runs": 1, "failure_rate": 0, "meets_minimum_10": False},
        {"workflow_id": "w1", "commit_sha": "c1", "success": 1, "failure": 0, "timed_out": 0, "eligible_runs": 1, "failure_rate": 0, "meets_minimum_10": False}
    ])
    with pytest.raises(AssertionError):
        validate_dataframe(bad_df2)
