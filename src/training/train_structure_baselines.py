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
from src.models.structure_baselines import StructureBaseline
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
    df = pd.read_parquet("data/processed/model_ready_structure_v1.parquet")
    
    # Read manifest to drop diagnostic-only
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
        
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    print(f"Dropping {len(diagnostic_features)} diagnostic-only features: {diagnostic_features}")
    
    feature_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split"] + diagnostic_features]
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    
    X_train = train_df[feature_cols]
    y_train = train_df["final_label"]
    X_val = val_df[feature_cols]
    y_val = val_df["final_label"]
    X_test = test_df[feature_cols]
    y_test = test_df["final_label"]
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Feature count: {len(feature_cols)}")
    
    start_time = time.time()
    
    model = StructureBaseline(
        model_params=config["model_parameters"],
        use_scaler=config.get("use_scaler", False)
    )
    
    print("\nFitting model on TRAIN ONLY...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    print("\nGenerating predictions...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    print("Evaluating...")
    train_metrics = evaluate_predictions(y_train, y_train_pred)
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)
    
    metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics
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
    
    # Feature Importance
    print("Extracting feature importance...")
    importance_df = None
    if hasattr(model.model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.model.feature_importances_
        }).sort_values("importance", ascending=False)
    elif hasattr(model.model, "coef_"):
        # For multi-class Logistic Regression, coef_ is (n_classes, n_features)
        # We can take the mean absolute coefficient across classes
        mean_abs_coef = np.mean(np.abs(model.model.coef_), axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": mean_abs_coef
        }).sort_values("importance", ascending=False)
        
    if importance_df is not None:
        importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
        print("Top 5 features:")
        print(importance_df.head(5).to_string(index=False))
        
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
    return val_metrics["macro_f1"], val_metrics["accuracy"], test_metrics["macro_f1"], test_metrics["accuracy"]

if __name__ == "__main__":
    v1 = run_experiment(Path("configs/experiments/e03_structure_logreg.yaml"))
    v2 = run_experiment(Path("configs/experiments/e04_structure_rf.yaml"))
    v3 = run_experiment(Path("configs/experiments/e05_structure_gbdt.yaml"))
    
    print("\n===============================")
    
    results = [
        ("E03", "Structure Logistic Regression", v1[0]),
        ("E04", "Structure Random Forest", v2[0]),
        ("E05", "Structure Gradient Boosting", v3[0])
    ]
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Winner (by Validation Macro F1):")
    print(f"{results[0][0]}: {results[0][1]} ({results[0][2]:.4f})")
    
    print("\n--- Summary Table for Report ---")
    
    # also read e01 and e02 from disk
    e01_path = Path("experiments/e01_tfidf_logreg/metrics.json")
    e02_path = Path("experiments/e02_tfidf_linear_svm/metrics.json")
    
    try:
        e01_metrics = json.loads(e01_path.read_text())
        e01_val_f1 = e01_metrics["validation"]["macro_f1"]
        e01_test_f1 = e01_metrics["test"]["macro_f1"]
        e01_val_acc = e01_metrics["validation"]["accuracy"]
        e01_test_acc = e01_metrics["test"]["accuracy"]
    except:
        e01_val_f1 = e01_test_f1 = e01_val_acc = e01_test_acc = 0.0
        
    try:
        e02_metrics = json.loads(e02_path.read_text())
        e02_val_f1 = e02_metrics["validation"]["macro_f1"]
        e02_test_f1 = e02_metrics["test"]["macro_f1"]
        e02_val_acc = e02_metrics["validation"]["accuracy"]
        e02_test_acc = e02_metrics["test"]["accuracy"]
    except:
        e02_val_f1 = e02_test_f1 = e02_val_acc = e02_test_acc = 0.0

    print("| Model | Representation | Validation Macro-F1 | Test Macro-F1 | Validation Accuracy | Test Accuracy |")
    print("|---|---|---|---|---|---|")
    print(f"| E01 TF-IDF Logistic Regression | Text | {e01_val_f1:.4f} | {e01_test_f1:.4f} | {e01_val_acc:.4f} | {e01_test_acc:.4f} |")
    print(f"| E02 TF-IDF Linear SVM | Text | {e02_val_f1:.4f} | {e02_test_f1:.4f} | {e02_val_acc:.4f} | {e02_test_acc:.4f} |")
    print(f"| E03 Structure Logistic Regression | Structure | {v1[0]:.4f} | {v1[2]:.4f} | {v1[1]:.4f} | {v1[3]:.4f} |")
    print(f"| E04 Structure Random Forest | Structure | {v2[0]:.4f} | {v2[2]:.4f} | {v2[1]:.4f} | {v2[3]:.4f} |")
    print(f"| E05 Structure Gradient Boosting | Structure | {v3[0]:.4f} | {v3[2]:.4f} | {v3[1]:.4f} | {v3[3]:.4f} |")
