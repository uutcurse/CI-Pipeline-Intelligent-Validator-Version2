import pytest
import os
import json
import yaml
import numpy as np
from pathlib import Path
from src.inference.workflow_preprocessor import WorkflowPreprocessor
from src.preprocessing.normalize_workflow import normalize_workflow
from src.features.structural_features import extract_workflow_features

@pytest.fixture(scope="module")
def preprocessor():
    return WorkflowPreprocessor()

def test_valid_yaml(preprocessor):
    yaml_in = "name: CI\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo 'hello'"
    pw = preprocessor.parse_yaml(yaml_in)
    assert pw["name"] == "CI"
    assert pw["job_count"] == 1
    assert pw["step_count"] == 1
    assert pw["jobs"]["test"]["runs_on"] == "ubuntu-latest" # Hyphen to underscore

def test_malformed_yaml(preprocessor):
    with pytest.raises(ValueError, match="Malformed YAML"):
        preprocessor.parse_yaml("name: CI\njobs: \n  test: \n    runs-on: [")

def test_empty_yaml(preprocessor):
    with pytest.raises(ValueError, match="Empty YAML"):
        preprocessor.parse_yaml("")

def test_invalid_workflow(preprocessor):
    with pytest.raises(ValueError, match="Invalid GitHub Actions workflow: 'jobs' mapping missing"):
        preprocessor.parse_yaml("name: CI\nenv: \n  FOO: bar")

def test_exact_80_feature_count(preprocessor):
    yaml_in = "name: CI\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo 'hello'"
    pw = preprocessor.parse_yaml(yaml_in)
    struct = preprocessor.extract_structural_features(pw)
    assert len(struct) == 80

def test_exact_feature_ordering(preprocessor):
    yaml_in = "name: CI\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo 'hello'"
    pw = preprocessor.parse_yaml(yaml_in)
    struct = preprocessor.extract_structural_features(pw)
    assert list(struct.keys()) == preprocessor.required_features

def test_forbidden_field_exclusion(preprocessor):
    yaml_in = "name: CI\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo 'hello'"
    pw = preprocessor.parse_yaml(yaml_in)
    struct = preprocessor.extract_structural_features(pw)
    # The output struct shouldn't magically contain 'repository' because extract_workflow_features doesn't put it there.
    assert "repository" not in struct

def test_integration_with_e06(preprocessor):
    yaml_in = "name: CI\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo 'hello'"
    res = preprocessor.predict(yaml_in)
    assert "prediction" in res
    assert "predicted_class" in res["prediction"]
    assert "preprocessing" in res
    assert res["preprocessing"]["feature_count"] == 80

def test_repeated_deterministic(preprocessor):
    yaml_in = "name: CI\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo 'hello'"
    res1 = preprocessor.predict(yaml_in)
    res2 = preprocessor.predict(yaml_in)
    assert res1["prediction"]["predicted_class"] == res2["prediction"]["predicted_class"]
    assert res1["prediction"]["probabilities"] == res2["prediction"]["probabilities"]

def test_golden_fixtures(preprocessor):
    fixtures_dir = Path("tests/fixtures/workflows")
    yml_files = list(fixtures_dir.glob("*.yml"))
    assert len(yml_files) >= 10
    
    for yml_file in yml_files:
        with open(yml_file, 'r') as f:
            yaml_in = f.read()
            
        json_file = yml_file.with_suffix('').with_name(yml_file.stem + "_truth.json")
        with open(json_file, 'r') as f:
            truth_pw = json.load(f)
            
        # Production Parsing
        prod_pw = preprocessor.parse_yaml(yaml_in)
        
        # Test Normalization Equivalence
        prod_text = preprocessor.generate_normalized_text(prod_pw)
        truth_text = normalize_workflow(truth_pw)
        assert prod_text == truth_text
        
        # Test Structural Features Equivalence
        prod_struct = preprocessor.extract_structural_features(prod_pw)
        
        # Get raw truth features
        truth_raw_features = extract_workflow_features(truth_pw)
        # Truth might have diagnostics, filter them
        for f_name in preprocessor.required_features:
            val_p = prod_struct[f_name]
            val_t = float(truth_raw_features.get(f_name, 0.0))
            assert np.isclose(val_p, val_t), f"Mismatch in {yml_file.name} feature {f_name}"
