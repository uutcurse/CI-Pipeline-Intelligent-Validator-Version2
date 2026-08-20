import json
import yaml
import time
import pandas as pd
from typing import Dict, Any, Union
from pathlib import Path

# Add src to path if needed for independent execution
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.normalize_workflow import normalize_workflow
from src.features.structural_features import extract_workflow_features
from src.inference.service import E06InferenceService

class WorkflowPreprocessor:
    def __init__(self, contract_path='data/manifests/final_model_input_contract_v1.json'):
        self.contract_path = Path(contract_path)
        if self.contract_path.exists():
            with open(self.contract_path, 'r') as f:
                self.contract = json.load(f)
            self.required_features = self.contract['inputs']['structure']['features']
        else:
            self.required_features = None
            
        self.service = None # Lazy load
        
    def parse_yaml(self, yaml_input: str) -> Dict[str, Any]:
        if not yaml_input or not str(yaml_input).strip():
            raise ValueError("Empty YAML input.")
            
        try:
            pw = yaml.safe_load(yaml_input)
        except yaml.YAMLError as e:
            raise ValueError(f"Malformed YAML: {e}")
            
        if not isinstance(pw, dict):
            raise ValueError("YAML root must be a mapping.")
            
        # Fix for YAML 1.1 evaluating unquoted 'on:' as boolean True
        if True in pw:
            pw["on"] = pw.pop(True)
            
        if "jobs" not in pw:
            raise ValueError("Invalid GitHub Actions workflow: 'jobs' mapping missing.")
            
        if not isinstance(pw.get("jobs"), dict):
            raise ValueError("Invalid GitHub Actions workflow: 'jobs' must be a mapping.")
            
        # Standardize hyphens to underscores as done in dataset creation
        self._standardize_keys(pw)
        
        # Inject metadata needed by normalize_workflow
        job_count = len(pw.get("jobs", {}))
        step_count = 0
        for j in pw.get("jobs", {}).values():
            if isinstance(j, dict) and isinstance(j.get("steps"), list):
                step_count += len(j["steps"])
                
        pw["job_count"] = job_count
        pw["step_count"] = step_count
        
        # We don't try to calculate languages, services, runners, actions_used 
        # at the root because they aren't critical for structural predictions
        # and aren't used by the normalization engine except for the meta block 
        # which can gracefully omit them.
        
        return pw
        
    def _standardize_keys(self, obj):
        """Recursively rename hyphenated GitHub Actions fields to underscored."""
        if not isinstance(obj, dict):
            return
            
        replacements = {
            "runs-on": "runs_on",
            "timeout-minutes": "timeout_minutes",
            "continue-on-error": "continue_on_error",
            "working-directory": "working_directory"
        }
        
        # Extract keys to list to allow mutating dictionary
        keys = list(obj.keys())
        for k in keys:
            v = obj[k]
            
            if k in replacements:
                obj[replacements[k]] = v
                del obj[k]
                k = replacements[k]
                
            if isinstance(v, dict):
                self._standardize_keys(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        self._standardize_keys(item)

    def generate_normalized_text(self, pw: Dict[str, Any]) -> str:
        return normalize_workflow(pw)
        
    def extract_structural_features(self, pw: Dict[str, Any]) -> Dict[str, float]:
        raw_features = extract_workflow_features(pw)
        
        if not self.required_features:
            raise RuntimeError("Input contract not loaded.")
            
        struct_features = {}
        for feat in self.required_features:
            if feat in raw_features:
                val = raw_features[feat]
                # Coerce to float to guarantee numeric output without NaNs
                struct_features[feat] = float(val) if val is not None else 0.0
            else:
                # If feature logic missed something (edge case), default to 0.0
                struct_features[feat] = 0.0
                
        return struct_features

    def prepare_model_input(self, yaml_input: str) -> Dict[str, Any]:
        pw = self.parse_yaml(yaml_input)
        text = self.generate_normalized_text(pw)
        struct = self.extract_structural_features(pw)
        
        return {
            "normalized_workflow_text": text,
            "structural_features": struct,
            "_parsed_workflow": pw # attached for metadata
        }
        
    def predict(self, yaml_input: str) -> Dict[str, Any]:
        if not self.service:
            self.service = E06InferenceService(contract_path=self.contract_path)
            
        model_input = self.prepare_model_input(yaml_input)
        
        start = time.time()
        pred = self.service.predict(
            model_input["normalized_workflow_text"], 
            model_input["structural_features"]
        )
        duration = time.time() - start
        
        pw = model_input["_parsed_workflow"]
        
        return {
            "prediction": pred,
            "preprocessing": {
                "workflow_name": pw.get("name"),
                "job_count": pw.get("job_count", 0),
                "step_count": pw.get("step_count", 0),
                "feature_count": len(model_input["structural_features"]),
                "structural_features": model_input["structural_features"]
            },
            "inference_duration": duration
        }
