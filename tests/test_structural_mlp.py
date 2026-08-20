import pytest
import torch
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')

from src.models.structural_mlp import StructuralMLP

def test_model_shapes_and_forward():
    model = StructuralMLP(input_dim=80)
    x = torch.randn(10, 80)
    
    # Test 3, 4, 5: model input/output shapes, embedding shape == 32, logits shape == (batch, 3)
    logits, emb = model(x, return_embedding=True)
    assert logits.shape == (10, 3)
    assert emb.shape == (10, 32)
    
    # Test 8: deterministic forward shape
    assert logits.shape == model(x).shape

def test_feature_dimensions_and_forbidden_columns():
    df = pd.read_parquet('data/processed/model_ready_structure_v1.parquet')
    
    exclude = ['sample_id', 'final_label', 'split',
               'permissions_none_count', 'language_count',
               'language_diversity', 'declared_language_presence']
    
    forbidden = ['repository', 'owner', 'repo', 'workflow_id', 'commit_sha', 'failure_rate', 'success', 'failure', 'timed_out', 'eligible_runs']
    
    features = [c for c in df.columns if c not in exclude]
    
    # Test 2: feature dimension is 80
    assert len(features) == 80
    
    # Test 6: no forbidden columns enter the model
    for f in forbidden:
        assert f not in features

def test_embedding_schema():
    if os.path.exists('data/intermediate/structural_embeddings_v1.parquet'):
        emb_df = pd.read_parquet('data/intermediate/structural_embeddings_v1.parquet')
        
        # Test 9: embedding export schema
        expected_cols = {'sample_id', 'split', 'final_label'} | {f'embedding_{i}' for i in range(32)}
        assert set(emb_df.columns) == expected_cols
        
        # Test 6: no forbidden columns in embeddings
        forbidden = ['repository', 'owner', 'repo', 'workflow_id', 'commit_sha', 'failure_rate', 'success', 'failure', 'timed_out', 'eligible_runs']
        for f in forbidden:
            assert f not in emb_df.columns

def test_checkpoint_save_load(tmp_path):
    model = StructuralMLP()
    path = tmp_path / "model.pt"
    
    # Test 7: checkpoint save/load
    torch.save(model.state_dict(), path)
    model2 = StructuralMLP()
    model2.load_state_dict(torch.load(path))
    
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
