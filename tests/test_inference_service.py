import pytest
import json
import numpy as np
import pandas as pd
from src.inference.service import E06InferenceService

@pytest.fixture(scope="module")
def service():
    return E06InferenceService()

def test_health_check(service):
    h = service.health()
    assert h["status"] == "UP"
    assert h["model_loaded"] is True
    assert h["contract_loaded"] is True

def test_model_info(service):
    info = service.model_info()
    assert info["model_id"] == "E06"
    assert info["feature_count"] == 80
    assert "LOW" in info["label_mapping"].values()

def test_reference_predictions(service):
    # Load 20 known samples
    test_preds = pd.read_parquet('experiments/e06_hybrid_logreg/final_test_predictions_v1.parquet')
    sample_preds = test_preds.head(20)
    
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    # Filter to matching rows
    df_sample = df[df['sample_id'].isin(sample_preds['sample_id'])]
    
    with open('data/manifests/final_model_input_contract_v1.json', 'r') as f:
        contract = json.load(f)
    features = contract['inputs']['structure']['features']
    
    match_count = 0
    for idx, pred_row in sample_preds.iterrows():
        sid = pred_row['sample_id']
        row = df_sample[df_sample['sample_id'] == sid].iloc[0]
        
        text = row['normalized_workflow_text']
        struct_dict = {f: float(row[f]) for f in features}
        
        res = service.predict(text, struct_dict)
        
        assert res["predicted_class"] == pred_row["predicted_label"]
        match_count += 1
        
    assert match_count == 20

def get_sample_input():
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    test_df = df[df['split'] == 'test'].iloc[0]
    with open('data/manifests/final_model_input_contract_v1.json', 'r') as f:
        contract = json.load(f)
    struct_dict = {f: float(test_df[f]) for f in contract['inputs']['structure']['features']}
    return str(test_df['normalized_workflow_text']), struct_dict

def test_valid_inference(service):
    text, struct = get_sample_input()
    res = service.predict(text, struct)
    assert "predicted_class" in res
    assert "predicted_label" in res
    assert "probabilities" in res
    
def test_probability_sum(service):
    text, struct = get_sample_input()
    res = service.predict(text, struct)
    probs = res["probabilities"]
    assert np.isclose(probs["LOW"] + probs["MEDIUM"] + probs["HIGH"], 1.0)
    
def test_deterministic_inference(service):
    text, struct = get_sample_input()
    res1 = service.predict(text, struct)
    res2 = service.predict(text, struct)
    assert res1["predicted_class"] == res2["predicted_class"]
    assert res1["probabilities"]["LOW"] == res2["probabilities"]["LOW"]

def test_missing_feature(service):
    text, struct = get_sample_input()
    del struct[list(struct.keys())[0]]
    with pytest.raises(ValueError, match="Required structural feature"):
        service.predict(text, struct)

def test_unexpected_feature(service):
    text, struct = get_sample_input()
    struct["unknown_feat"] = 1.0
    with pytest.raises(ValueError, match="Unexpected structural features"):
        service.predict(text, struct)

def test_non_finite_feature(service):
    text, struct = get_sample_input()
    struct[list(struct.keys())[0]] = np.inf
    with pytest.raises(ValueError, match="must be a finite number"):
        service.predict(text, struct)

def test_empty_text(service):
    _, struct = get_sample_input()
    with pytest.raises(ValueError, match="Input text is missing or empty"):
        service.predict("   ", struct)

def test_forbidden_field(service):
    text, struct = get_sample_input()
    struct["repository"] = "a/b"
    with pytest.raises(ValueError, match="Forbidden field"):
        service.predict(text, struct)
