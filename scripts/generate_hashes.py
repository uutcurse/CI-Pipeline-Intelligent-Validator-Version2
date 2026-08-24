import os
import hashlib
import json
import sklearn
import xgboost
import pandas
import numpy
import joblib
import platform

def hash_file(path):
    if not os.path.exists(path): return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

files_to_hash = {
    "Production": {
        "model": "experiments/e06_hybrid_logreg/model.joblib"
    },
    "Research": {
        "model": "experiments/final_candidate/model.joblib",
        "calibrated_model": "experiments/final_candidate/calibrated_model.joblib",
        "scaler": "experiments/final_candidate/scaler.joblib",
        "tfidf": "experiments/final_candidate/tfidf.joblib",
        "svd": "experiments/final_candidate/svd.joblib"
    }
}

manifest = {
    "hashes": {
        "Production": {k: hash_file(v) for k, v in files_to_hash["Production"].items()},
        "Research": {k: hash_file(v) for k, v in files_to_hash["Research"].items()}
    },
    "environment": {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__
    }
}

with open("experiments/final_candidate/artifact_manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)
