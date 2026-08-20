import json
import logging
import warnings
import time
import uuid
import numpy as np
import pandas as pd
from pathlib import Path

# Configure basic logging
logger = logging.getLogger(__name__)

class E06InferenceService:
    def __init__(
        self, 
        model_dir='experiments/e06_hybrid_logreg', 
        contract_path='data/manifests/final_model_input_contract_v1.json'
    ):
        self.model_dir = Path(model_dir)
        self.contract_path = Path(contract_path)
        self._model = None
        self._contract = None
        
        self._load_artifacts()

    def _load_artifacts(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        if not self.contract_path.exists():
            raise FileNotFoundError(f"Contract not found: {self.contract_path}")
            
        with open(self.contract_path, 'r') as f:
            self._contract = json.load(f)
            
        self.struct_features = self._contract['inputs']['structure']['features']
        self.forbidden_fields = set(self._contract['forbidden_fields'])
        self.label_mapping = self._contract['target']['mapping']
        
        # Load the frozen model exactly once
        import joblib
        from src.models.hybrid_classical import HybridBaseline
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._model = HybridBaseline.load(self.model_dir / 'model.joblib')
            
        logger.info(f"Loaded E06 model from {self.model_dir}")

    def health(self):
        """Verify model and contract are loaded and accessible."""
        return {
            "status": "UP" if self._model is not None and self._contract is not None else "DOWN",
            "model_loaded": self._model is not None,
            "contract_loaded": self._contract is not None
        }

    def model_info(self):
        """Return model metadata from the contract."""
        if not self._contract:
            return {}
        return {
            "model_id": self._contract.get("model_id"),
            "model_version": self._contract.get("contract_version"),
            "feature_count": self._contract["inputs"]["structure"].get("length"),
            "model_type": self._contract.get("model"),
            "label_mapping": self._contract["target"]["mapping"]
        }

    def predict(self, text: str, struct_dict: dict):
        start_time = time.time()
        req_id = str(uuid.uuid4())
        
        # 1. Validation
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text is missing or empty.")
            
        if struct_dict is None or not isinstance(struct_dict, dict):
            raise ValueError("Structural features must be a dictionary.")
            
        for f in self.forbidden_fields:
            if f in struct_dict:
                raise ValueError(f"Forbidden field '{f}' is present in input.")
                
        struct_vec = []
        for feat in self.struct_features:
            if feat not in struct_dict:
                raise ValueError(f"Required structural feature '{feat}' is missing.")
            val = struct_dict[feat]
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                raise ValueError(f"Feature '{feat}' must be a finite number.")
            struct_vec.append(val)
            
        if len(struct_dict) != len(self.struct_features):
            unexpected = set(struct_dict.keys()) - set(self.struct_features)
            if unexpected:
                raise ValueError(f"Unexpected structural features present: {unexpected}")

        # 2. Representation Construction
        X_text = pd.Series([text])
        X_struct = pd.DataFrame([struct_vec], columns=self.struct_features)
        
        # 3. Inference (No fitting)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_trans = self._model.transform(X_text, X_struct)
            pred_class = int(self._model.model.predict(X_trans)[0])
            probs = self._model.model.predict_proba(X_trans)[0]
            
        # 4. Output formatting
        pred_label = self.label_mapping.get(str(pred_class))
        prob_dict = {
            self.label_mapping["0"]: float(probs[0]),
            self.label_mapping["1"]: float(probs[1]),
            self.label_mapping["2"]: float(probs[2])
        }
        
        duration = time.time() - start_time
        logger.info(f"req_id={req_id} duration={duration:.4f}s pred={pred_class} version={self._contract.get('contract_version')}")
        
        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "probabilities": prob_dict
        }

