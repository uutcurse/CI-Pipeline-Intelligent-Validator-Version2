import sys
import os
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def check_file(path):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        sys.exit(1)

def hash_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

print("Verifying critical files...")
check_file('experiments/e06_hybrid_logreg/model.joblib')
check_file('data/manifests/final_model_input_contract_v1.json')
check_file('data/manifests/final_model_manifest_v1.json')
check_file('data/manifests/final_system_validation_manifest_v1.json')

h = hash_file('experiments/e06_hybrid_logreg/model.joblib')
if h != '155d9d0ca1f2e9d73f096176449de1e31bce1c9ec4f3e9e6fc85882b10234519':
    print(f"HASH MISMATCH for model.joblib: {h}")
    sys.exit(1)

print("Verifying API & Preprocessor Imports...")
try:
    from src.inference.service import E06InferenceService
    from src.inference.workflow_preprocessor import WorkflowPreprocessor
    from src.inference.predict import E06InferencePipeline
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

print("Verifying E06 Inference sample prediction...")
yaml_sample = '''
name: Test
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo hello
'''
try:
    pp = WorkflowPreprocessor()
    res = pp.predict(yaml_sample)
    if 'LOW' not in res['prediction']['probabilities']:
        print("MISSING PREDICTION KEYS")
        sys.exit(1)
except Exception as e:
    print(f"PREDICTION ERROR: {e}")
    sys.exit(1)

print("Reproducibility Check: PASS")
sys.exit(0)
