import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.text_baselines import TFIDFBaseline

def test_tfidf_baseline():
    vec_params = {
        "ngram_range": (1, 1),
        "min_df": 1,
        "max_df": 1.0,
        "sublinear_tf": True
    }
    mod_params = {
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42
    }
    
    model = TFIDFBaseline(vec_params, mod_params.copy())
    
    train_texts = ["hello world", "goodbye world", "hello again", "python test"]
    train_labels = [0, 1, 0, 2]
    
    test_texts = ["hello python", "goodbye"]
    
    # 1. Fit only on training split
    X_train_vec = model.fit(train_texts, train_labels)
    assert X_train_vec.shape[0] == 4
    
    # 2. Validation/test transform does not refit vocabulary
    vocab = set(model.vectorizer.vocabulary_.keys())
    assert "hello" in vocab
    assert "python" in vocab
    
    X_test_vec = model.transform(test_texts)
    assert X_test_vec.shape[0] == 2
    assert X_test_vec.shape[1] == len(vocab)
    
    # Vocabulary should remain unchanged
    assert set(model.vectorizer.vocabulary_.keys()) == vocab
    
    # 5. Model prediction shape
    preds = model.predict(test_texts)
    assert preds.shape == (2,)
    
    # 4. Label mapping / 6. Probability output
    probs = model.predict_proba(test_texts)
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    # 7. Deterministic
    model2 = TFIDFBaseline(vec_params, mod_params.copy())
    model2.fit(train_texts, train_labels)
    preds2 = model2.predict(test_texts)
    assert np.array_equal(preds, preds2)
