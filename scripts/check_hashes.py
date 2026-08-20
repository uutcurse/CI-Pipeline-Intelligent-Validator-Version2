import hashlib
import json
import os

files_to_hash = [
    'experiments/e06_hybrid_logreg/model.joblib',
    'data/manifests/final_model_input_contract_v1.json',
    'data/manifests/final_model_manifest_v1.json',
    'data/manifests/final_model_hashes_v1.json'
]

hashes = {}
for f in files_to_hash:
    if os.path.exists(f):
        with open(f, 'rb') as file:
            hashes[f] = hashlib.sha256(file.read()).hexdigest()
    else:
        hashes[f] = None

print(json.dumps(hashes, indent=2))
