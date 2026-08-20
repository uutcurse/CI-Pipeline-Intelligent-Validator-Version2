import pytest
import pandas as pd
from pathlib import Path

def test_model_ready_dataset_integrity():
    text_path = Path("data/processed/model_ready_text_v1.parquet")
    struct_path = Path("data/processed/model_ready_structure_v1.parquet")
    hybrid_path = Path("data/processed/model_ready_hybrid_v1.parquet")
    prov_path = Path("data/processed/model_provenance_v1.parquet")
    
    if not text_path.exists():
        pytest.skip("Data not generated yet")
        
    df_text = pd.read_parquet(text_path)
    df_struct = pd.read_parquet(struct_path)
    df_hybrid = pd.read_parquet(hybrid_path)
    df_prov = pd.read_parquet(prov_path)
    
    # Row counts
    assert len(df_text) == 12944
    assert len(df_struct) == 12944
    assert len(df_hybrid) == 12944
    assert len(df_prov) == 12944
    
    # Sample ID uniqueness
    assert df_text["sample_id"].is_unique
    assert df_struct["sample_id"].is_unique
    assert df_hybrid["sample_id"].is_unique
    assert df_prov["sample_id"].is_unique
    
    # Label preservation
    assert set(df_text["final_label"].unique()).issubset({0, 1, 2})
    assert len(set(df_text["final_label"].unique())) == 3
    
    # Split preservation
    assert set(df_text["split"].unique()).issubset({"train", "validation", "test"})
    assert len(set(df_text["split"].unique())) == 3
    
    # No forbidden features in views
    forbidden = ["failure_rate", "success", "failure", "timed_out", "eligible_runs"]
    for f in forbidden:
        assert f not in df_text.columns
        assert f not in df_struct.columns
        assert f not in df_hybrid.columns
        
    # No missing text
    assert not df_text["normalized_workflow_text"].isna().any()
    
    # Matching indices (sample_id aligns)
    assert set(df_text["sample_id"]) == set(df_struct["sample_id"])
    assert set(df_hybrid["sample_id"]) == set(df_prov["sample_id"])
