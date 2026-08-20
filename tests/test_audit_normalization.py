import pytest
import pandas as pd
import json

from src.preprocessing.audit_normalization import deterministic_hash, canonical_json_hash

def test_deterministic_hashing():
    s1 = "hello world"
    s2 = "hello world"
    assert deterministic_hash(s1) == deterministic_hash(s2)
    assert deterministic_hash(s1) == 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'

def test_canonical_json_hashing():
    # different order, same content
    j1 = '{"a": 1, "b": 2}'
    j2 = '{"b": 2, "a": 1}'
    assert canonical_json_hash(j1) == canonical_json_hash(j2)
    
    # different content
    j3 = '{"a": 1, "b": 3}'
    assert canonical_json_hash(j1) != canonical_json_hash(j3)

def test_exact_duplicate_detection():
    df = pd.DataFrame({
        "normalized_workflow_text": ["A", "B", "A", "C"],
        "split": ["train", "test", "train", "validation"]
    })
    df["norm_hash"] = df["normalized_workflow_text"].apply(deterministic_hash)
    group_sizes = df.groupby("norm_hash").size()
    duplicates = group_sizes[group_sizes > 1]
    assert len(duplicates) == 1
    assert duplicates.iloc[0] == 2

def test_cross_split_leakage():
    df = pd.DataFrame({
        "normalized_workflow_text": ["A", "A", "B", "B"],
        "split": ["train", "validation", "train", "train"]
    })
    df["norm_hash"] = df["normalized_workflow_text"].apply(deterministic_hash)
    
    cross_split = 0
    for h, size in df.groupby("norm_hash").size().items():
        if size > 1:
            splits = df[df["norm_hash"] == h]["split"].unique()
            if len(splits) > 1:
                cross_split += 1
                
    assert cross_split == 1

def test_raw_vs_normalized_collision():
    df = pd.DataFrame({
        "parsed_workflow_json": ['{"a": 1}', '{"a": 2}', '{"a": 3}'],
        "normalized_workflow_text": ["NormA", "NormA", "NormB"]
    })
    
    df["norm_hash"] = df["normalized_workflow_text"].apply(deterministic_hash)
    df["raw_hash"] = df["parsed_workflow_json"].apply(canonical_json_hash)
    
    collisions = 0
    for h, size in df.groupby("norm_hash").size().items():
        if size > 1:
            raw_hashes = df[df["norm_hash"] == h]["raw_hash"].unique()
            if len(raw_hashes) > 1:
                collisions += 1
                
    assert collisions == 1
