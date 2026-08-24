import json
import joblib
import pandas as pd
import numpy as np

def main():
    print("Loading artifacts...")
    try:
        model = joblib.load("experiments/final_candidate/model.joblib")
        calib_model = joblib.load("experiments/final_candidate/calibrated_model.joblib")
        scaler = joblib.load("experiments/final_candidate/scaler.joblib")
        tfidf = joblib.load("experiments/final_candidate/tfidf.joblib")
        svd = joblib.load("experiments/final_candidate/svd.joblib")
    except Exception as e:
        print(f"FAILED to load artifacts: {e}")
        exit(1)
        
    print("Loading test sample...")
    df = pd.read_parquet("data/processed/model_ready_hybrid_v1.parquet").iloc[[0]]
    
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
    diagnostic = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", "normalized_workflow_text"] + diagnostic]
    
    X_text = df["normalized_workflow_text"].values
    X_struct = df[struct_cols].values
    
    print("Preprocessing...")
    X_tfidf = tfidf.transform(X_text)
    X_svd = svd.transform(X_tfidf)
    X_st = scaler.transform(X_struct)
    X_hybrid = np.hstack([X_svd, X_st])
    
    print("Predicting...")
    # Uncalibrated class
    pred = model.predict(X_hybrid)[0]
    
    # Calibrated prob
    prob = calib_model.predict_proba(X_hybrid)[0]
    confidence = np.max(prob)
    
    classes = ["LOW", "MEDIUM", "HIGH"]
    pred_class = classes[pred]
    
    print("=== SMOKE TEST SUCCESS ===")
    print(f"Predicted Class: {pred_class} (Raw index: {pred})")
    print(f"Probabilities: {prob}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()
