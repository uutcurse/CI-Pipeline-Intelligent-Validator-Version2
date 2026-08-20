import json
import yaml
import sys
import time

manifest = {
    "normalization_spec_version": "v1",
    "structural_feature_version": "v1",
    "feature_count": 80,
    "feature_order_hash": "inherited_from_final_model_input_contract_v1",
    "model_input_contract_version": "v1",
    "parser_library": "PyYAML",
    "parser_version": yaml.__version__,
    "software_versions": {
        "python": sys.version
    },
    "timestamp": time.time()
}

with open("data/manifests/production_preprocessing_manifest_v1.json", "w") as f:
    json.dump(manifest, f, indent=4)
