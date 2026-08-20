import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib
import warnings
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.text_baselines import TFIDFBaseline
from src.models.structure_baselines import StructureBaseline
from src.models.hybrid_classical import HybridBaseline
from src.evaluation.classification_metrics import evaluate_predictions

def get_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return h

def main():
    out_dir = Path("experiments/robustness_classical_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123, 2024, 3407, 7777]
    
    print("Loading data...")
    df_hybrid = pd.read_parquet("data/processed/model_ready_hybrid_v1.parquet")
    
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
        
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    redundant_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "REDUNDANT_CANDIDATE"]
    
    text_col = "normalized_workflow_text"
    struct_cols = [c for c in df_hybrid.columns if c not in ["sample_id", "final_label", "split", text_col] + diagnostic_features]
    
    train_df = df_hybrid[df_hybrid["split"] == "train"]
    val_df = df_hybrid[df_hybrid["split"] == "validation"]
    test_df = df_hybrid[df_hybrid["split"] == "test"]
    
    train_texts = set(train_df[text_col])
    test_novel_mask = ~test_df[text_col].isin(train_texts)
    
    results = []
    e06_no_redundant_val_f1 = []
    
    for seed in seeds:
        print(f"\n======================================")
        print(f"Running seed: {seed}")
        print(f"======================================")
        
        with open("configs/experiments/e02_tfidf_linear_svm.yaml") as f:
            c_e02 = yaml.safe_load(f)
        with open("configs/experiments/e05_structure_gbdt.yaml") as f:
            c_e05 = yaml.safe_load(f)
        with open("configs/experiments/e06_hybrid_logreg.yaml") as f:
            c_e06 = yaml.safe_load(f)
        
        # --- E02 ---
        c_e02["model_parameters"]["random_state"] = seed
        model_e02 = TFIDFBaseline(c_e02["vectorizer_parameters"], c_e02["model_parameters"])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_e02.fit(train_df[text_col], train_df["final_label"])
            
        y_val_pred_e02 = model_e02.predict(val_df[text_col])
        y_test_pred_e02 = model_e02.predict(test_df[text_col])
        val_m_e02 = evaluate_predictions(val_df["final_label"], y_val_pred_e02)
        test_m_e02 = evaluate_predictions(test_df["final_label"], y_test_pred_e02)
        
        # --- E05 ---
        c_e05["model_parameters"]["random_state"] = seed
        model_e05 = StructureBaseline(c_e05["model_parameters"], use_scaler=c_e05.get("use_scaler", False))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_e05.fit(train_df[struct_cols], train_df["final_label"])
            
        y_val_pred_e05 = model_e05.predict(val_df[struct_cols])
        y_test_pred_e05 = model_e05.predict(test_df[struct_cols])
        val_m_e05 = evaluate_predictions(val_df["final_label"], y_val_pred_e05)
        test_m_e05 = evaluate_predictions(test_df["final_label"], y_test_pred_e05)
        
        # --- E06 ---
        c_e06["model_parameters"]["random_state"] = seed
        model_e06 = HybridBaseline(c_e06["vectorizer_parameters"], c_e06["model_parameters"], use_scaler=c_e06.get("use_scaler", True))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_e06.fit(train_df[text_col], train_df[struct_cols], train_df["final_label"])
            
        y_val_pred_e06 = model_e06.predict(val_df[text_col], val_df[struct_cols])
        y_test_pred_e06 = model_e06.predict(test_df[text_col], test_df[struct_cols])
        val_m_e06 = evaluate_predictions(val_df["final_label"], y_val_pred_e06)
        test_m_e06 = evaluate_predictions(test_df["final_label"], y_test_pred_e06)
        
        # Strict novel eval for E06
        y_test_strict_pred_e06 = y_test_pred_e06[test_novel_mask]
        test_strict_m_e06 = evaluate_predictions(test_df["final_label"][test_novel_mask], y_test_strict_pred_e06)
        
        # --- E06 No Redundant ---
        struct_cols_no_redundant = [c for c in struct_cols if c not in redundant_features]
        # Reinitialize params otherwise dict gets popped
        c_e06_tmp = yaml.safe_load(open("configs/experiments/e06_hybrid_logreg.yaml"))
        c_e06_tmp["model_parameters"]["random_state"] = seed
        model_e06_nr = HybridBaseline(c_e06_tmp["vectorizer_parameters"], c_e06_tmp["model_parameters"], use_scaler=c_e06_tmp.get("use_scaler", True))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_e06_nr.fit(train_df[text_col], train_df[struct_cols_no_redundant], train_df["final_label"])
        y_val_pred_e06_nr = model_e06_nr.predict(val_df[text_col], val_df[struct_cols_no_redundant])
        val_m_e06_nr = evaluate_predictions(val_df["final_label"], y_val_pred_e06_nr)
        e06_no_redundant_val_f1.append(val_m_e06_nr["macro_f1"])
        
        # Store results
        def store_model_res(model_id, val_m, test_m, strict_test_f1=None):
            row = {
                "seed": seed,
                "model": model_id,
                "val_macro_f1": val_m["macro_f1"],
                "val_balanced_accuracy": val_m["balanced_accuracy"],
                "test_macro_f1": test_m["macro_f1"],
                "test_balanced_accuracy": test_m["balanced_accuracy"],
                "test_accuracy": test_m["accuracy"],
                "test_class0_f1": test_m["per_class_f1"]["0"],
                "test_class1_f1": test_m["per_class_f1"]["1"],
                "test_class2_f1": test_m["per_class_f1"]["2"],
            }
            if strict_test_f1 is not None:
                row["test_strict_novel_macro_f1"] = strict_test_f1
            results.append(row)
            
        store_model_res("E02", val_m_e02, test_m_e02)
        store_model_res("E05", val_m_e05, test_m_e05)
        store_model_res("E06", val_m_e06, test_m_e06, test_strict_m_e06["macro_f1"])

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    
    # Calculate Summaries
    summary = []
    for m in ["E02", "E05", "E06"]:
        sub = res_df[res_df["model"] == m]
        ci_val = get_ci(sub["val_macro_f1"])
        ci_test = get_ci(sub["test_macro_f1"])
        
        row = {
            "model": m,
            "val_macro_f1_mean": sub["val_macro_f1"].mean(),
            "val_macro_f1_std": sub["val_macro_f1"].std(),
            "val_macro_f1_min": sub["val_macro_f1"].min(),
            "val_macro_f1_max": sub["val_macro_f1"].max(),
            "val_macro_f1_ci95": ci_val,
            "test_macro_f1_mean": sub["test_macro_f1"].mean(),
            "test_macro_f1_std": sub["test_macro_f1"].std(),
            "test_macro_f1_min": sub["test_macro_f1"].min(),
            "test_macro_f1_max": sub["test_macro_f1"].max(),
            "test_macro_f1_ci95": ci_test,
        }
        if m == "E06":
            row["test_strict_novel_macro_f1_mean"] = sub["test_strict_novel_macro_f1"].mean()
            row["test_strict_novel_macro_f1_std"] = sub["test_strict_novel_macro_f1"].std()
        
        summary.append(row)
        
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    
    # Calculate E06 - E02 and E06 - E05
    diffs = []
    for seed in seeds:
        s_df = res_df[res_df["seed"] == seed]
        e02_f1 = s_df[s_df["model"] == "E02"]["test_macro_f1"].values[0]
        e05_f1 = s_df[s_df["model"] == "E05"]["test_macro_f1"].values[0]
        e06_f1 = s_df[s_df["model"] == "E06"]["test_macro_f1"].values[0]
        
        diffs.append({
            "seed": seed,
            "E06_minus_E02": e06_f1 - e02_f1,
            "E06_minus_E05": e06_f1 - e05_f1
        })
    diff_df = pd.DataFrame(diffs)
    diff_df.to_csv(out_dir / "seed_comparison.csv", index=False)
    
    # Report Markdown Generation
    md = ["# Classical Baselines Robustness Audit\n"]
    md.append("## Summary Statistics\n")
    md.append(sum_df.to_markdown(index=False) + "\n")
    
    md.append("## Pairwise Differences (Test Macro F1)\n")
    md.append(diff_df.to_markdown(index=False) + "\n")
    
    mean_diff_text = diff_df["E06_minus_E02"].mean()
    mean_diff_struct = diff_df["E06_minus_E05"].mean()
    md.append(f"\nMean E06 vs E02: {mean_diff_text:+.4f}")
    md.append(f"\nMean E06 vs E05: {mean_diff_struct:+.4f}\n")
    
    md.append("## Strict Novel-Configuration Robustness (E06)\n")
    e06_strict = res_df[res_df["model"] == "E06"]["test_strict_novel_macro_f1"]
    md.append(f"Mean Strict Test F1: {e06_strict.mean():.4f} ± {get_ci(e06_strict):.4f}\n")
    
    md.append("## Redundancy Candidates Comparison\n")
    e06_base = res_df[res_df["model"] == "E06"]["val_macro_f1"]
    md.append(f"E06 All Features (Val F1 Mean): {e06_base.mean():.4f}")
    md.append(f"\nE06 Without Redundant Candidates (Val F1 Mean): {np.mean(e06_no_redundant_val_f1):.4f}\n")
    
    Path("reports/classical_robustness_v1.md").write_text("\n".join(md))
    
    # Dump Config
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "seeds": seeds,
            "models_evaluated": ["E02", "E05", "E06"],
            "timestamp": datetime.datetime.utcnow().isoformat()
        }, f, indent=4)
        
    print("\nRobustness Audit Complete.")

if __name__ == "__main__":
    main()
