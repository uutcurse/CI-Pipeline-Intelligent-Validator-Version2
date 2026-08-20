import json
import pandas as pd

with open('data/manifests/structural_feature_manifest_v1.json', 'r') as f:
    manifest = json.load(f)
diag = [f['feature_name'] for f in manifest['features'] if f['status'] == 'DIAGNOSTIC_ONLY']

df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
struct_cols = [c for c in df.columns if c not in ['sample_id', 'final_label', 'split', 'normalized_workflow_text'] + diag]

contract = {
    "contract_version": "1.0",
    "model_id": "E06",
    "inputs": {
        "text": {
            "name": "normalized_workflow_text",
            "type": "string",
            "description": "Textual representation of the GitHub Actions workflow."
        },
        "structure": {
            "type": "vector",
            "length": 80,
            "features": struct_cols
        }
    },
    "transforms": {
        "text": "TfidfVectorizer (frozen)",
        "structure": "StandardScaler (frozen)"
    },
    "model": "LogisticRegression (scikit-learn)",
    "target": {
        "name": "final_label",
        "type": "integer",
        "classes": [0, 1, 2],
        "mapping": {
            "0": "LOW",
            "1": "MEDIUM",
            "2": "HIGH"
        }
    },
    "forbidden_fields": [
        "repository", "owner", "repo", "workflow_id", "workflow_path", 
        "commit_sha", "commit_date", "file_sha", "split", "success", 
        "failure", "timed_out", "eligible_runs", "failure_rate", 
        "meets_minimum_10", "provisional_class"
    ]
}

with open('data/manifests/final_model_input_contract_v1.json', 'w') as f:
    json.dump(contract, f, indent=4)
