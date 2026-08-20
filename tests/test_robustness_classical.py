import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.robustness_classical import get_ci
from src.models.hybrid_classical import HybridBaseline

def test_confidence_interval():
    data = [0.5, 0.6, 0.7, 0.55, 0.65]
    ci = get_ci(data)
    assert ci > 0
    assert ci < 0.2

def test_seed_propagation():
    model1 = HybridBaseline(
        vectorizer_params={"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0},
        model_params={"model_type": "LogisticRegression", "random_state": 42}
    )
    model2 = HybridBaseline(
        vectorizer_params={"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0},
        model_params={"model_type": "LogisticRegression", "random_state": 123}
    )
    
    # Just asserting the random_state propagated correctly to the internal model
    assert model1.model.random_state == 42
    assert model2.model.random_state == 123
