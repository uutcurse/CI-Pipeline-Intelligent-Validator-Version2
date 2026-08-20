import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.build_labelled_corpus import compute_repo_split, calculate_final_thresholds

def test_repository_cap_logic():
    df = pd.DataFrame({
        "repository": ["repo1"] * 25 + ["repo2"] * 5,
        "workflow_id": list(range(30)),
        "commit_date": ["2020-01-01"] * 30,
        "commit_sha": [str(i) for i in range(30)],
        "parsing_success": True,
        "parsed_workflow_available": True
    })
    
    df["_grp"] = df["repository"]
    df_cap = df.groupby("_grp", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 20), random_state=42)
    )
    df_cap.drop(columns=["_grp"], inplace=True, errors="ignore")
    
    assert len(df_cap[df_cap["repository"] == "repo1"]) == 20
    assert len(df_cap[df_cap["repository"] == "repo2"]) == 5
    assert len(df_cap) == 25

def test_repository_split_leakage():
    df = pd.DataFrame({
        "repository": [f"repo{i}" for i in range(100)],
        "failure_rate": np.random.rand(100)
    })
    # Expand to multiple versions
    df = pd.concat([df]*3).reset_index(drop=True)
    
    df_split, train_r, val_r, test_r = compute_repo_split(df)
    
    # Check no leakage
    assert len(train_r & val_r) == 0
    assert len(train_r & test_r) == 0
    assert len(val_r & test_r) == 0
    
    # Check completeness
    assert len(train_r | val_r | test_r) == 100
    
    # Check proportions (approx 70, 15, 15)
    assert 65 <= len(train_r) <= 75
    assert 10 <= len(val_r) <= 20
    assert 10 <= len(test_r) <= 20

def test_threshold_calculation_leakage_protection():
    # If validation data changes, thresholds MUST NOT change
    df1 = pd.DataFrame({
        "repository": ["repo_t", "repo_t", "repo_v"],
        "split": ["train", "train", "validation"],
        "failure_rate": [0.1, 0.9, 0.5]
    })
    
    df2 = pd.DataFrame({
        "repository": ["repo_t", "repo_t", "repo_v"],
        "split": ["train", "train", "validation"],
        "failure_rate": [0.1, 0.9, 0.0]  # Validation data changed!
    })
    
    _, t1_1, t2_1 = calculate_final_thresholds(df1)
    _, t1_2, t2_2 = calculate_final_thresholds(df2)
    
    assert t1_1 == t1_2, "T1 changed when validation data changed!"
    assert t2_1 == t2_2, "T2 changed when validation data changed!"

def test_exact_boundary_handling():
    df = pd.DataFrame({
        "split": ["train", "train", "train", "train", "train"],
        "failure_rate": [0.0, 0.25, 0.5, 0.75, 1.0]
    })
    df_out, t1, t2 = calculate_final_thresholds(df)
    
    assert df_out[df_out["failure_rate"] == 0.0]["final_label"].iloc[0] == 0
    assert df_out[df_out["failure_rate"] == 0.25]["final_label"].iloc[0] == 0
    assert df_out[df_out["failure_rate"] == 0.5]["final_label"].iloc[0] == 1
    assert df_out[df_out["failure_rate"] == 0.75]["final_label"].iloc[0] == 2
    assert df_out[df_out["failure_rate"] == 1.0]["final_label"].iloc[0] == 2
