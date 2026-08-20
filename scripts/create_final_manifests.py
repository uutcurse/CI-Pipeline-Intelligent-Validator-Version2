import hashlib
import json
import sys
import sklearn
import numpy
import pandas
import random

def file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

hashes = {
    "model_artifact": file_hash('experiments/e06_hybrid_logreg/model.joblib'),
    "input_contract": file_hash('data/manifests/final_model_input_contract_v1.json'),
    "final_label_definition": file_hash('data/manifests/final_label_definition_v1.json'),
    "repository_split_manifest": file_hash('data/manifests/repository_split_v1.json'),
    "model_comparison_table": file_hash('reports/final_model_comparison_v1.csv')
}

with open('data/manifests/final_model_hashes_v1.json', 'w') as f:
    json.dump(hashes, f, indent=4)

manifest = {
    "model_id": "E06",
    "model_name": "HybridBaseline_LogisticRegression",
    "dataset_version": "v1",
    "label_definition_version": "v1",
    "split_version": "v1",
    "feature_version": "v1",
    "normalization_version": "v1",
    "vectorizer_version": "TfidfVectorizer(max_features=250000)",
    "scaler_version": "StandardScaler",
    "model_parameters": {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced"
    },
    "software_versions": {
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__
    },
    "random_seed": 42,
    "artifact_hashes": hashes,
    "test_metrics": {
        "macro_f1": 0.49718068330698895,
        "accuracy": 0.49950298210735584,
        "balanced_accuracy": 0.4988775619376674
    }
}

with open('data/manifests/final_model_manifest_v1.json', 'w') as f:
    json.dump(manifest, f, indent=4)
