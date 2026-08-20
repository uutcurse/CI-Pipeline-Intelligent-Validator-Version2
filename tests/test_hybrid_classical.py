import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.hybrid_classical import HybridBaseline

def test_hybrid_fitting_and_transform():
    model = HybridBaseline(
        vectorizer_params={"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0},
        model_params={"model_type": "LogisticRegression"},
        use_scaler=True
    )
    
    X_text_train = ["hello world", "goodbye"]
    X_struct_train = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
    y_train = [0, 1]
    
    X_hybrid = model.fit(X_text_train, X_struct_train, y_train)
    
    # Vocabulary should have 3 words: 'hello', 'world', 'goodbye'
    assert len(model.vectorizer.vocabulary_) == 3
    # X_hybrid should be sparse matrix with 2 rows, 5 columns (3 text + 2 struct)
    assert X_hybrid.shape == (2, 5)
    
    # Check that transform does not change vocab or scaler
    X_text_test = ["hello python", "goodbye world"]
    X_struct_test = pd.DataFrame({"a": [10, 20], "b": [40, 50]})
    
    X_hybrid_test = model.transform(X_text_test, X_struct_test)
    assert len(model.vectorizer.vocabulary_) == 3
    assert X_hybrid_test.shape == (2, 5)
    
    # Output dims
    preds = model.predict(X_text_test, X_struct_test)
    assert preds.shape == (2,)
