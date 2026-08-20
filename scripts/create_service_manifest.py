import json
import time
import sys

manifest = {
    "model_id": "E06",
    "artifact_path": "experiments/e06_hybrid_logreg/model.joblib",
    "input_contract": "data/manifests/final_model_input_contract_v1.json",
    "feature_count": 80,
    "label_mapping": {
        "0": "LOW",
        "1": "MEDIUM",
        "2": "HIGH"
    },
    "software_version": {
        "python": sys.version
    },
    "timestamp": time.time()
}

with open("data/manifests/inference_service_manifest_v1.json", "w") as f:
    json.dump(manifest, f, indent=4)
