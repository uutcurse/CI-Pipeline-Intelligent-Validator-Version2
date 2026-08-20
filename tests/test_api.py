import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "UP"
        assert data["model_loaded"] is True
        assert data["preprocessor_loaded"] is True

def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "E06"
        assert data["feature_count"] == 80
        assert data["frozen"] is True

def test_predict_valid_workflow():
    workflow_yaml = '''
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo 'hello'
'''
    with TestClient(app) as client:
        response = client.post("/predict", json={"workflow_yaml": workflow_yaml})
        assert response.status_code == 200
        data = response.json()
        
        # Check schema
        assert "prediction" in data
        assert "workflow" in data
        assert "inference_duration_ms" in data
        
        # Check prediction structure
        pred = data["prediction"]
        assert "predicted_class" in pred
        assert "predicted_label" in pred
        assert "probabilities" in pred
        
        # Check probabilities
        probs = pred["probabilities"]
        assert "LOW" in probs
        assert "MEDIUM" in probs
        assert "HIGH" in probs
        
        # Probability sum
        total_prob = probs["LOW"] + probs["MEDIUM"] + probs["HIGH"]
        assert np.isclose(total_prob, 1.0)
        
        # Check no sensitive data
        assert "repository" not in data["workflow"]
        
        # Check request ID
        assert "x-request-id" in response.headers

def test_predict_empty_workflow():
    with TestClient(app) as client:
        response = client.post("/predict", json={"workflow_yaml": ""})
        assert response.status_code == 422 # Pydantic min_length=1 Validation

def test_predict_malformed_yaml():
    with TestClient(app) as client:
        response = client.post("/predict", json={"workflow_yaml": "name: [\njobs:"})
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == "INVALID_WORKFLOW"

def test_predict_invalid_workflow():
    with TestClient(app) as client:
        response = client.post("/predict", json={"workflow_yaml": "name: CI"})
        assert response.status_code == 400
        data = response.json()
        assert "jobs" in data["error"]["message"]

def test_deterministic_prediction():
    workflow_yaml = '''
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo 'hello'
'''
    with TestClient(app) as client:
        res1 = client.post("/predict", json={"workflow_yaml": workflow_yaml}).json()
        res2 = client.post("/predict", json={"workflow_yaml": workflow_yaml}).json()
        
        assert res1["prediction"] == res2["prediction"]

def test_oversized_request():
    with TestClient(app) as client:
        # 1MB limit by default. Send 1.5MB of data
        large_yaml = "name: CI\njobs:\n" + ("  " * (1024 * 1024))
        # Add content-length manually to trigger early failure middleware
        response = client.post("/predict", json={"workflow_yaml": large_yaml}, headers={"content-length": str(len(large_yaml.encode('utf-8')) + 100)})
        assert response.status_code == 413

def test_cors_allowed_origin_5173():
    with TestClient(app) as client:
        response = client.options("/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

def test_cors_allowed_origin_127():
    with TestClient(app) as client:
        response = client.options("/predict", headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"

def test_cors_disallowed_origin():
    with TestClient(app) as client:
        response = client.options("/predict", headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 400
        # The allowed origin shouldn't be the evil one
        assert response.headers.get("access-control-allow-origin") is None
