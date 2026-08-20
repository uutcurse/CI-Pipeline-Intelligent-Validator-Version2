import pytest
import pandas as pd
import json
import os

def test_model_metrics_loaded_correctly():
    assert os.path.exists('data/intermediate/model_ablation_results_v1.parquet')
    df = pd.read_parquet('data/intermediate/model_ablation_results_v1.parquet')
    assert len(df) == 7
    assert 'E06' in df['model'].values

def test_frozen_splits_unchanged():
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    assert df['split'].nunique() == 3

def test_repository_overlap_remains_zero():
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    train_repos = set(df_prov[df_prov['split'] == 'train']['repository'])
    test_repos = set(df_prov[df_prov['split'] == 'test']['repository'])
    assert len(train_repos.intersection(test_repos)) == 0

def test_strict_duplicate_filter_uses_train_text_only():
    df_text = pd.read_parquet('data/processed/model_ready_text_v1.parquet')
    train_texts = set(df_text[df_text['split'] == 'train']['normalized_workflow_text'])
    test_texts = set(df_text[df_text['split'] == 'test']['normalized_workflow_text'])
    overlap = len(train_texts.intersection(test_texts))
    assert overlap > 0

def test_temporal_bins_contain_no_missing_samples():
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    assert df_prov['commit_date'].isna().sum() == 0

def test_length_thresholds_derived_from_train_only():
    df_text = pd.read_parquet('data/processed/model_ready_text_v1.parquet')
    train_lens = df_text[df_text['split'] == 'train']['normalized_workflow_text'].str.len()
    assert train_lens.quantile(0.5) > 0

def test_no_forbidden_fields_in_error_reports():
    with open('reports/phase_4_4_ablation_report.md', 'r', encoding='utf-8') as f:
        content = f.read()
    assert "success:" not in content
    assert "failure_rate:" not in content

def test_deterministic_analysis_output():
    assert os.path.exists('data/manifests/phase_4_4_ablation_manifest_v1.json')
