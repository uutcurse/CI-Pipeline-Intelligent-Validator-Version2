from typing import Dict, List
import json
import os
import pickle
from datetime import datetime
from unittest import result
import numpy as np
import pandas as pd


# =========================================================
# PATH SETUP (portable, no Kaggle dependency)
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# =========================================================
# CORE DATA STRUCTURES
# =========================================================

class ValidationResult:
    """Container for a single file validation result"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.timestamp = datetime.now().isoformat()
        self.syntax_valid = False
        self.schema_valid = False
        self.content_ok = False
        self.ml_risk_score = 0.0
        self.final_risk_score = 0.0
        self.risk_level = "UNKNOWN"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.details: Dict = {}

    def is_safe(self) -> bool:
        return self.risk_level in ["LOW", "MEDIUM"]

    def to_dict(self) -> Dict:
        return {
            "filepath": self.filepath,
            "timestamp": self.timestamp,
            "syntax_valid": self.syntax_valid,
            "schema_valid": self.schema_valid,
            "content_ok": self.content_ok,
            "ml_risk_score": self.ml_risk_score,
            "final_risk_score": self.final_risk_score,
            "risk_level": self.risk_level,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


# =========================================================
# VALIDATORS
# =========================================================

class JSONSyntaxValidator:
    def validate(self, content: str):
        try:
            json.loads(content)
            return True, []
        except json.JSONDecodeError as e:
            return False, [f"JSON syntax error: {str(e)}"]


class JSONSchemaValidator:
    def __init__(self, schema: Dict):
        self.schema = schema

    def validate(self, data: Dict):
        try:
            from jsonschema import validate
            validate(instance=data, schema=self.schema)
            return True, []
        except Exception as e:
            return False, [str(e)]


class JSONContentAnalyzer:
    def analyze(self, data: Dict) -> Dict:
        return {
            "complexity_score": len(str(data)) / 1000,
            "has_secrets_keywords": False,
            "suspicious_patterns": [],
        }


# =========================================================
# FEATURE EXTRACTION & ML
# =========================================================

class FeatureExtractor:
    def extract(self, data: Dict, filepath: str) -> Dict[str, float]:
        return {
            "file_size_kb": os.path.getsize(filepath) / 1024 if os.path.exists(filepath) else 0,
            "num_keys": len(data.keys()) if isinstance(data, dict) else 0,
            "depth": self._get_depth(data),
        }

    def _get_depth(self, obj, depth=1):
        if isinstance(obj, dict):
            return max((self._get_depth(v, depth + 1) for v in obj.values()), default=depth)
        if isinstance(obj, list):
            return max((self._get_depth(i, depth + 1) for i in obj), default=depth)
        return depth


class MLRiskPredictor:
    def __init__(self, models_dir: str):
        with open(os.path.join(models_dir, "best_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

    def predict_risk(self, features: Dict[str, float]) -> float:
        df = pd.DataFrame([features])
        X_scaled = self.scaler.transform(df)
        return float(self.model.predict_proba(X_scaled)[0][1])


class RiskScoringEngine:
    def __init__(self):
        self.thresholds = {"LOW": 0.3, "MEDIUM": 0.7}

    def compute(self, syntax_ok: bool, schema_ok: bool, content_ok: bool, ml_risk: float):
        rule_penalty = 0.0
        if not syntax_ok:
            rule_penalty += 0.4
        if not schema_ok:
            rule_penalty += 0.4
        if not content_ok:
            rule_penalty += 0.2

        combined = min(1.0, rule_penalty + ml_risk)

        if combined < self.thresholds["LOW"]:
            level = "LOW"
        elif combined < self.thresholds["MEDIUM"]:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return round(combined * 100, 2), level


# =========================================================
# MAIN BRAIN CLASS
# =========================================================

class CIPipelineValidator:
    """Unified CI Pipeline Validator (THE BRAIN)"""

    def __init__(self):
        config_path = os.path.join(BASE_DIR, "config.json")
        schema_path = os.path.join(BASE_DIR, "pipeline_schema.json")
        models_dir = os.path.join(BASE_DIR, "models")

        with open(schema_path, "r") as f:
            schema = json.load(f)

        self.syntax_validator = JSONSyntaxValidator()
        self.schema_validator = JSONSchemaValidator(schema)
        self.content_analyzer = JSONContentAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.ml_predictor = MLRiskPredictor(models_dir)
        self.risk_scorer = RiskScoringEngine()

        self.stats = {
            "files_validated": 0,
            "files_passed": 0,
            "files_failed": 0,
            "high_risk_count": 0,
        }

    def validate_file(self, filepath: str) -> ValidationResult:
        result = ValidationResult(filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                data = json.loads(content)

            # Syntax
            syntax_ok, syntax_errors = self.syntax_validator.validate(content)
            result.syntax_valid = syntax_ok
            result.errors.extend(syntax_errors)

            # Schema
            schema_ok = False
            if syntax_ok:
                schema_ok, schema_errors = self.schema_validator.validate(data)
                result.schema_valid = schema_ok
                result.errors.extend(schema_errors)

            # Content
            content_info = self.content_analyzer.analyze(data)
            result.content_ok = not content_info["has_secrets_keywords"]
            result.details["content_analysis"] = content_info

            # ML
            features = self.feature_extractor.extract(data, filepath)
            try:
                ml_risk = self.ml_predictor.predict_risk(features)
            except Exception as e:
                ml_risk = 0.0
                result.warnings.append("ML model skipped due to feature mismatch")

            result.ml_risk_score = ml_risk

            # Final scoring
            score, level = self.risk_scorer.compute(
                syntax_ok, schema_ok, result.content_ok, ml_risk
            )
            result.final_risk_score = score
            result.risk_level = level

            # Stats
            self.stats["files_validated"] += 1
            if level in ["LOW", "MEDIUM"]:
                self.stats["files_passed"] += 1
            else:
                self.stats["files_failed"] += 1
                self.stats["high_risk_count"] += 1

        except Exception as e:
            result.errors.append(str(e))
            result.risk_level = "HIGH"
            self.stats["files_failed"] += 1
            self.stats["high_risk_count"] += 1

        return result

    # âœ… API / Backend friendly wrapper
    def validate_file_dict(self, filepath: str) -> Dict:
        return self.validate_file(filepath).to_dict()

    def get_stats(self) -> Dict:
        return self.stats.copy()


