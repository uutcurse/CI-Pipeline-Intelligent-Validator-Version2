import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Suppress scikit-learn version warnings during load
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class E06InferencePipeline:
    def __init__(self, model_dir='experiments/e06_hybrid_logreg', contract_path='data/manifests/final_model_input_contract_v1.json'):
        import joblib
        from src.models.hybrid_classical import HybridBaseline
        
        self.model_dir = Path(model_dir)
        self.contract_path = Path(contract_path)
        
        if not self.model_dir.exists() or not self.contract_path.exists():
            raise FileNotFoundError("Model or contract not found.")
            
        with open(self.contract_path, 'r') as f:
            self.contract = json.load(f)
            
        self.struct_features = self.contract['inputs']['structure']['features']
        self.forbidden_fields = set(self.contract['forbidden_fields'])
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.pipeline = HybridBaseline.load(self.model_dir / 'model.joblib')

    def predict(self, text: str, struct_dict: dict):
        # Validation
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text is missing or empty.")
            
        for f in self.forbidden_fields:
            if f in struct_dict:
                raise ValueError(f"Forbidden field '{f}' is present in input.")
                
        # Ensure all required structural features exist and build vector in EXACT order
        struct_vec = []
        for feat in self.struct_features:
            if feat not in struct_dict:
                raise ValueError(f"Required structural feature '{feat}' is missing.")
            val = struct_dict[feat]
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                raise ValueError(f"Feature '{feat}' must be a finite number.")
            struct_vec.append(val)
            
        # Check for unexpected features silently substituted
        if len(struct_dict) != len(self.struct_features):
            unexpected = set(struct_dict.keys()) - set(self.struct_features)
            if unexpected:
                raise ValueError(f"Unexpected structural features present: {unexpected}")

        X_text = pd.Series([text])
        X_struct = pd.DataFrame([struct_vec], columns=self.struct_features)
        
        # Inference (no fitting allowed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_trans = self.pipeline.transform(X_text, X_struct)
            pred = self.pipeline.model.predict(X_trans)[0]
            probs = self.pipeline.model.predict_proba(X_trans)[0]
            
        return {
            "predicted_class": int(pred),
            "probabilities": probs.tolist()
        }
