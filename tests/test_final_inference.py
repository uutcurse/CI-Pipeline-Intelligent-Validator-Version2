import pytest
import json
import numpy as np
import pandas as pd
from src.inference.predict import E06InferencePipeline

@pytest.fixture(scope="module")
def pipeline():
    return E06InferencePipeline()

@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    test_df = df[df['split'] == 'test'].iloc[0]
    
    with open('data/manifests/final_model_input_contract_v1.json', 'r') as f:
        contract = json.load(f)
    
    struct_dict = {f: float(test_df[f]) for f in contract['inputs']['structure']['features']}
    return str(test_df['normalized_workflow_text']), struct_dict, int(test_df['sample_id'].split('_')[1]) # Just to identify if needed, but not part of dict

def test_valid_sample_produces_prediction(pipeline, sample_data):
    text, struct, _ = sample_data
    res = pipeline.predict(text, struct)
    assert 'predicted_class' in res
    assert 'probabilities' in res

def test_prediction_belongs_to_012(pipeline, sample_data):
    text, struct, _ = sample_data
    res = pipeline.predict(text, struct)
    assert res['predicted_class'] in [0, 1, 2]

def test_probability_vector(pipeline, sample_data):
    text, struct, _ = sample_data
    res = pipeline.predict(text, struct)
    probs = res['probabilities']
    assert len(probs) == 3
    assert np.isclose(sum(probs), 1.0)

def test_deterministic_output(pipeline, sample_data):
    text, struct, _ = sample_data
    res1 = pipeline.predict(text, struct)
    res2 = pipeline.predict(text, struct)
    assert res1['predicted_class'] == res2['predicted_class']
    assert np.allclose(res1['probabilities'], res2['probabilities'])

def test_missing_text_raises(pipeline, sample_data):
    _, struct, _ = sample_data
    with pytest.raises(ValueError, match="Input text is missing or empty"):
        pipeline.predict("", struct)

def test_missing_feature_raises(pipeline, sample_data):
    text, struct, _ = sample_data
    bad_struct = struct.copy()
    del bad_struct[list(bad_struct.keys())[0]]
    with pytest.raises(ValueError, match="Required structural feature"):
        pipeline.predict(text, bad_struct)

def test_unexpected_feature_raises(pipeline, sample_data):
    text, struct, _ = sample_data
    bad_struct = struct.copy()
    bad_struct['some_random_feature'] = 1.0
    with pytest.raises(ValueError, match="Unexpected structural features"):
        pipeline.predict(text, bad_struct)

def test_forbidden_field_raises(pipeline, sample_data):
    text, struct, _ = sample_data
    bad_struct = struct.copy()
    bad_struct['repository'] = "foo/bar"
    # Even if we remove a valid one to keep length same, 'repository' trigger should fire first
    with pytest.raises(ValueError, match="Forbidden field 'repository' is present"):
        pipeline.predict(text, bad_struct)

def test_non_finite_feature_raises(pipeline, sample_data):
    text, struct, _ = sample_data
    bad_struct = struct.copy()
    bad_struct[list(bad_struct.keys())[0]] = np.nan
    with pytest.raises(ValueError, match="must be a finite number"):
        pipeline.predict(text, bad_struct)

