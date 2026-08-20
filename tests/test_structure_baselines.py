import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.structure_baselines import StructureBaseline

def test_scaler_fits_on_train_only():
    X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    
    X_test = pd.DataFrame({"a": [10, 20], "b": [40, 50]})
    
    model = StructureBaseline({"model_type": "LogisticRegression"}, use_scaler=True)
    model.fit(X_train, y_train)
    
    # Check scaler means
    assert np.allclose(model.scaler.mean_, [2.0, 5.0])
    
    # transform test, ensure it uses train means
    X_test_scaled = model.transform(X_test)
    assert np.allclose(X_test_scaled[:, 0], (X_test["a"] - 2.0) / model.scaler.scale_[0])
    
def test_transformed_feature_dimensions():
    X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    
    model = StructureBaseline({"model_type": "RandomForestClassifier", "n_estimators": 2}, use_scaler=False)
    X_transformed = model.fit(X_train, y_train)
    assert X_transformed.shape == (3, 2)
    
def test_model_output_dimensions():
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    y_train = pd.Series([0, 1, 2, 0])
    
    X_test = pd.DataFrame({"a": [10, 20], "b": [40, 50]})
    
    model = StructureBaseline({"model_type": "LogisticRegression"})
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    assert preds.shape == (2,)
    
    probs = model.predict_proba(X_test)
    assert probs.shape == (2, 3)
