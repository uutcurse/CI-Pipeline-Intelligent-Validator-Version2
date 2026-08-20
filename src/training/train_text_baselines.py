import sys
import yaml
import json
import datetime
import pandas as pd
from pathlib import Path
import time
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.text_baselines import TFIDFBaseline
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
    df = pd.read_parquet("data/processed/model_ready_text_v1.parquet")
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Class distribution
    print("\nClass Distribution:")
    print("Train:", train_df["final_label"].value_counts().to_dict())
    print("Val:", val_df["final_label"].value_counts().to_dict())
    print("Test:", test_df["final_label"].value_counts().to_dict())
    
    start_time = time.time()
    
    model = TFIDFBaseline(
        vectorizer_params=config["vectorizer_parameters"],
        model_params=config["model_parameters"]
    )
    
    print("\nFitting model on TRAIN ONLY...")
    # TF-IDF fit strictly on train
    X_train_vec = model.fit(train_df["normalized_workflow_text"], train_df["final_label"])
    
    print("Transforming Val/Test...")
    X_val_vec = model.transform(val_df["normalized_workflow_text"])
    X_test_vec = model.transform(test_df["normalized_workflow_text"])
    
    vocab_size = len(model.vectorizer.vocabulary_)
    
    print("\nData Shapes:")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Train Matrix: {X_train_vec.shape}")
    print(f"Val Matrix: {X_val_vec.shape}")
    print(f"Test Matrix: {X_test_vec.shape}")
    
    print("\nGenerating predictions...")
    y_train_pred = model.model.predict(X_train_vec)
    y_val_pred = model.predict(val_df["normalized_workflow_text"])
    y_test_pred = model.predict(test_df["normalized_workflow_text"])
    
    print("Evaluating...")
    train_metrics = evaluate_predictions(train_df["final_label"], y_train_pred)
    val_metrics = evaluate_predictions(val_df["final_label"], y_val_pred)
    test_metrics = evaluate_predictions(test_df["final_label"], y_test_pred)
    
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
    cm_val = save_confusion_matrix_plot(
        val_df["final_label"], y_val_pred, 
        f"reports/figures/{exp_id}_confusion_matrix_validation.png",
        title=f"{exp_id} Validation CM"
    )
    
    cm_test = save_confusion_matrix_plot(
        test_df["final_label"], y_test_pred, 
        f"reports/figures/{exp_id}_confusion_matrix_test.png",
        title=f"{exp_id} Test CM"
    )
    
    pd.DataFrame(cm_test, columns=["Pred_0", "Pred_1", "Pred_2"], index=["True_0", "True_1", "True_2"]).to_csv(out_dir / "confusion_matrix.csv")
    
    end_time = time.time()
    
    # Save artifact
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
    return val_metrics["macro_f1"]

if __name__ == "__main__":
    v1 = run_experiment(Path("configs/experiments/e01_tfidf_logreg.yaml"))
    v2 = run_experiment(Path("configs/experiments/e02_tfidf_linear_svm.yaml"))
    
    print("\n===============================")
    print(f"Winner (by Validation Macro F1):")
    if v1 > v2:
        print("E01: TF-IDF + Logistic Regression")
    else:
        print("E02: TF-IDF + Linear SVM")
