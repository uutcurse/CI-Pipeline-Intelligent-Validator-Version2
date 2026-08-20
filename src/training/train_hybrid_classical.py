import sys
import yaml
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.hybrid_classical import HybridBaseline
from src.evaluation.classification_metrics import evaluate_predictions, save_confusion_matrix_plot

def run_experiment(config_path):
    print(f"\n======================================")
    print(f"Running experiment: {config_path.name}")
    print(f"======================================")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    exp_id = config["experiment_id"]
    out_dir = Path(f"experiments/{exp_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading data...")
    df = pd.read_parquet("data/processed/model_ready_hybrid_v1.parquet")
    
    # Read manifest to drop diagnostic-only
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
        
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    
    # Text col
    text_col = "normalized_workflow_text"
    
    # Struct cols
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", text_col] + diagnostic_features]
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    
    X_train_text = train_df[text_col]
    X_train_struct = train_df[struct_cols]
    y_train = train_df["final_label"]
    
    X_val_text = val_df[text_col]
    X_val_struct = val_df[struct_cols]
    y_val = val_df["final_label"]
    
    X_test_text = test_df[text_col]
    X_test_struct = test_df[struct_cols]
    y_test = test_df["final_label"]
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Structural Feature count: {len(struct_cols)}")
    
    start_time = time.time()
    
    model = HybridBaseline(
        vectorizer_params=config["vectorizer_parameters"],
        model_params=config["model_parameters"],
        use_scaler=config.get("use_scaler", True)
    )
    
    print("\nFitting model on TRAIN ONLY...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_hybrid_train = model.fit(X_train_text, X_train_struct, y_train)
    
    vocab_size = len(model.vectorizer.vocabulary_)
    print(f"TF-IDF features: {vocab_size}")
    print(f"Structural features: {len(struct_cols)}")
    print(f"Hybrid feature count: {X_hybrid_train.shape[1]}")
    
    # Matrix shapes
    X_hybrid_val = model.transform(X_val_text, X_val_struct)
    X_hybrid_test = model.transform(X_test_text, X_test_struct)
    print(f"Train Matrix: {X_hybrid_train.shape} (Sparsity: {X_hybrid_train.nnz / (X_hybrid_train.shape[0]*X_hybrid_train.shape[1]):.4f})")
    print(f"Val Matrix: {X_hybrid_val.shape}")
    print(f"Test Matrix: {X_hybrid_test.shape}")
    
    print("\nGenerating predictions...")
    y_train_pred = model.model.predict(X_hybrid_train)
    y_val_pred = model.model.predict(X_hybrid_val)
    y_test_pred = model.model.predict(X_hybrid_test)
    
    print("Evaluating...")
    train_metrics = evaluate_predictions(y_train, y_train_pred)
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)
    
    # Strict diagnostic test
    train_texts = set(train_df[text_col])
    test_novel_mask = ~test_df[text_col].isin(train_texts)
    removed_test_rows = (~test_novel_mask).sum()
    remaining_test_rows = test_novel_mask.sum()
    
    y_test_strict = y_test[test_novel_mask]
    y_test_pred_strict = y_test_pred[test_novel_mask]
    test_strict_metrics = evaluate_predictions(y_test_strict, y_test_pred_strict)
    
    metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "strict_test": {
            "metrics": test_strict_metrics,
            "removed_rows": int(removed_test_rows),
            "remaining_rows": int(remaining_test_rows)
        }
    }
    
    out_dir.joinpath("metrics.json").write_text(json.dumps(metrics, indent=4))
    
    print("\nVALIDATION METRICS:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    
    # Save Confusion Matrices
    save_confusion_matrix_plot(
        y_val, y_val_pred, 
        f"reports/figures/{exp_id}_confusion_matrix_validation.png",
        title=f"{exp_id} Validation CM"
    )
    
    cm_test = save_confusion_matrix_plot(
        y_test, y_test_pred, 
        f"reports/figures/{exp_id}_confusion_matrix_test.png",
        title=f"{exp_id} Test CM"
    )
    
    pd.DataFrame(cm_test, columns=["Pred_0", "Pred_1", "Pred_2"], index=["True_0", "True_1", "True_2"]).to_csv(out_dir / "confusion_matrix.csv")
        
    end_time = time.time()
    
    print(f"Saving artifacts to {out_dir}...")
    model.save(out_dir / "model.joblib")
    
    import sklearn
    full_config = {
        **config,
        "training_start": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "training_end": datetime.datetime.fromtimestamp(end_time).isoformat(),
        "runtime_seconds": end_time - start_time,
        "software_versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__
        },
        "best_validation_metric": val_metrics["macro_f1"],
        "test_metrics": test_metrics
    }
    out_dir.joinpath("config.json").write_text(json.dumps(full_config, indent=4))
    
    print(f"Experiment {exp_id} finished in {end_time - start_time:.1f}s.")
    return (
        val_metrics["macro_f1"], 
        val_metrics["accuracy"], 
        test_metrics["macro_f1"], 
        test_metrics["accuracy"],
        test_strict_metrics["macro_f1"],
        int(removed_test_rows),
        int(remaining_test_rows)
    )

if __name__ == "__main__":
    v6 = run_experiment(Path("configs/experiments/e06_hybrid_logreg.yaml"))
    v7 = run_experiment(Path("configs/experiments/e07_hybrid_linear_svm.yaml"))
    
    print("\n===============================")
    
    results = [
        ("E06", "Hybrid Logistic Regression", v6),
        ("E07", "Hybrid Linear SVM", v7)
    ]
    results.sort(key=lambda x: x[2][0], reverse=True)
    
    best_id, best_name, best_v = results[0]
    
    print(f"Winner (by Validation Macro F1):")
    print(f"{best_id}: {best_name} ({best_v[0]:.4f})")
    
    # Load past results
    def load_metrics(exp):
        p = Path(f"experiments/{exp}/metrics.json")
        if not p.exists(): return 0.0, 0.0, 0.0, 0.0
        m = json.loads(p.read_text())
        return m["validation"]["macro_f1"], m["test"]["macro_f1"], m["validation"]["accuracy"], m["test"]["accuracy"]
        
    e01 = load_metrics("e01_tfidf_logreg")
    e02 = load_metrics("e02_tfidf_linear_svm")
    e03 = load_metrics("e03_structure_logreg")
    e04 = load_metrics("e04_structure_rf")
    e05 = load_metrics("e05_structure_gbdt")
    
    print("\n--- Summary Table for Report ---")
    print("| Model | Representation | Validation Macro-F1 | Test Macro-F1 | Validation Accuracy | Test Accuracy |")
    print("|---|---|---|---|---|---|")
    print(f"| E01 TF-IDF Logistic Regression | Text | {e01[0]:.4f} | {e01[1]:.4f} | {e01[2]:.4f} | {e01[3]:.4f} |")
    print(f"| E02 TF-IDF Linear SVM | Text | {e02[0]:.4f} | {e02[1]:.4f} | {e02[2]:.4f} | {e02[3]:.4f} |")
    print(f"| E03 Structure Logistic Regression | Structure | {e03[0]:.4f} | {e03[1]:.4f} | {e03[2]:.4f} | {e03[3]:.4f} |")
    print(f"| E04 Structure Random Forest | Structure | {e04[0]:.4f} | {e04[1]:.4f} | {e04[2]:.4f} | {e04[3]:.4f} |")
    print(f"| E05 Structure Gradient Boosting | Structure | {e05[0]:.4f} | {e05[1]:.4f} | {e05[2]:.4f} | {e05[3]:.4f} |")
    print(f"| E06 Hybrid Logistic Regression | Hybrid | {v6[0]:.4f} | {v6[2]:.4f} | {v6[1]:.4f} | {v6[3]:.4f} |")
    print(f"| E07 Hybrid Linear SVM | Hybrid | {v7[0]:.4f} | {v7[2]:.4f} | {v7[1]:.4f} | {v7[3]:.4f} |")
    
    best_text_val_f1 = max(e01[0], e02[0])
    best_text_test_f1 = max(e01[1], e02[1])
    
    best_struct_val_f1 = max(e03[0], e04[0], e05[0])
    best_struct_test_f1 = max(e03[1], e04[1], e05[1])
    
    hybrid_val_f1 = best_v[0]
    hybrid_test_f1 = best_v[2]
    
    print("\n--- Incremental Gain ---")
    print(f"Hybrid vs Text (Val):   {hybrid_val_f1 - best_text_val_f1:+.4f}")
    print(f"Hybrid vs Text (Test):  {hybrid_test_f1 - best_text_test_f1:+.4f}")
    print(f"Hybrid vs Struct (Val): {hybrid_val_f1 - best_struct_val_f1:+.4f}")
    print(f"Hybrid vs Struct (Test):{hybrid_test_f1 - best_struct_test_f1:+.4f}")
    
    print("\n--- Exact-Duplicate Robustness ---")
    print(f"Removed Test Rows: {best_v[5]}")
    print(f"Remaining Test Rows: {best_v[6]}")
    print(f"Strict Novel-Config Test F1: {best_v[4]:.4f}")
