import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_classical import HybridBaseline

def macro_f1(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]

def bootstrap_ci(y_true, y_pred_model, y_pred_baseline, n_iterations=10000, seed=42):
    np.random.seed(seed)
    n = len(y_true)
    deltas = []
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n, n)
        y_true_boot = y_true[indices]
        y_pred_m_boot = y_pred_model[indices]
        y_pred_b_boot = y_pred_baseline[indices]
        
        f1_m = macro_f1(y_true_boot, y_pred_m_boot)
        f1_b = macro_f1(y_true_boot, y_pred_b_boot)
        deltas.append(f1_m - f1_b)
        
    deltas = np.array(deltas)
    lower = np.percentile(deltas, 2.5)
    upper = np.percentile(deltas, 97.5)
    return deltas, lower, upper

def holm_bonferroni(p_values):
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    m = len(p_values)
    adjusted_p = np.zeros(m)
    
    for i in range(m):
        adjusted_p[i] = sorted_p[i] * (m - i)
        
    # Enforce monotonicity (cumulative max)
    adjusted_p = np.maximum.accumulate(adjusted_p)
    # Bound by 1.0
    adjusted_p = np.minimum(adjusted_p, 1.0)
    
    # Reorder back to original indices
    final_adj = np.zeros(m)
    final_adj[sorted_indices] = adjusted_p
    return final_adj

def main():
    print("Loading data...")
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    # Text Vectorizer & Scaler setup
    text_col = "normalized_workflow_text"
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", text_col] + diagnostic_features]
    
    X_train_text, X_train_struct, y_train = train_df[text_col], train_df[struct_cols], train_df['final_label'].values
    X_test_text, X_test_struct, y_test = test_df[text_col], test_df[struct_cols], test_df['final_label'].values
    
    print("Fitting preprocessing & baselines...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, strip_accents='unicode')
    X_train_text_v = vectorizer.fit_transform(X_train_text)
    X_test_text_v = vectorizer.transform(X_test_text)
    
    scaler = StandardScaler()
    X_train_struct_s = scaler.fit_transform(X_train_struct)
    X_test_struct_s = scaler.transform(X_test_struct)
    
    # RF Structural (B3)
    m_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    m_rf.fit(X_train_struct_s, y_train)
    y_pred_rf = m_rf.predict(X_test_struct_s)
    
    # LogReg Struct (B1)
    m_lr_struct = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42)
    m_lr_struct.fit(X_train_struct_s, y_train)
    y_pred_lr_struct = m_lr_struct.predict(X_test_struct_s)
    
    # LogReg Text (B2)
    m_lr_text = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42)
    m_lr_text.fit(X_train_text_v, y_train)
    y_pred_lr_text = m_lr_text.predict(X_test_text_v)
    
    # E06 Hybrid (B5)
    e06_model = HybridBaseline.load("experiments/e06_hybrid_logreg/model.joblib")
    y_pred_e06 = e06_model.predict(X_test_text, X_test_struct)
    
    # Verification
    f1_e06 = macro_f1(y_test, y_pred_e06)
    f1_rf = macro_f1(y_test, y_pred_rf)
    f1_lr_struct = macro_f1(y_test, y_pred_lr_struct)
    f1_lr_text = macro_f1(y_test, y_pred_lr_text)
    
    print(f"E06: {f1_e06:.4f}")
    print(f"RF Struct: {f1_rf:.4f}")
    print(f"LR Struct: {f1_lr_struct:.4f}")
    print(f"LR Text: {f1_lr_text:.4f}")
    
    if abs(f1_e06 - 0.4972) > 0.001 or abs(f1_rf - 0.4818) > 0.001 or abs(f1_lr_struct - 0.4738) > 0.001 or abs(f1_lr_text - 0.4723) > 0.001:
        print("ERROR: Baseline F1 mismatch. Stopping.")
        sys.exit(1)
        
    out_dir = Path("experiments/statistical_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Paired Bootstrap...")
    np.random.seed(42)
    deltas_rf, ci_l_rf, ci_u_rf = bootstrap_ci(y_test, y_pred_e06, y_pred_rf)
    deltas_lrs, ci_l_lrs, ci_u_lrs = bootstrap_ci(y_test, y_pred_e06, y_pred_lr_struct)
    deltas_lrt, ci_l_lrt, ci_u_lrt = bootstrap_ci(y_test, y_pred_e06, y_pred_lr_text)
    
    # Save bootstrap dists
    boot_df = pd.DataFrame({
        "E06_vs_RF": deltas_rf,
        "E06_vs_LR_Struct": deltas_lrs,
        "E06_vs_LR_Text": deltas_lrt
    })
    boot_df.to_csv(out_dir / "bootstrap_distributions.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(deltas_rf, color='blue', label='E06 vs RF', kde=True, stat='density', alpha=0.5)
    plt.axvline(0, color='red', linestyle='--')
    plt.axvline(ci_l_rf, color='blue', linestyle=':')
    plt.axvline(ci_u_rf, color='blue', linestyle=':')
    plt.title('Bootstrap Distribution: Delta Macro F1 (E06 vs Random Forest Structural)')
    plt.xlabel('Delta Macro F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bootstrap_delta_e06_vs_rf.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(deltas_rf, color='blue', label='E06 vs RF (Struct)', fill=True, alpha=0.3)
    sns.kdeplot(deltas_lrs, color='green', label='E06 vs LR (Struct)', fill=True, alpha=0.3)
    sns.kdeplot(deltas_lrt, color='purple', label='E06 vs LR (Text)', fill=True, alpha=0.3)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Bootstrap Distributions: Delta Macro F1 (All Models)')
    plt.xlabel('Delta Macro F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bootstrap_delta_all_models.png")
    plt.close()
    
    print("Running Paired Permutation Test...")
    def run_permutation(y_true, y_p1, y_p2, n_perms=10000, seed=42):
        np.random.seed(seed)
        n = len(y_true)
        obs_diff = macro_f1(y_true, y_p1) - macro_f1(y_true, y_p2)
        count = 0
        for _ in range(n_perms):
            swaps = np.random.randint(0, 2, n).astype(bool)
            y_p1_perm = np.where(swaps, y_p2, y_p1)
            y_p2_perm = np.where(swaps, y_p1, y_p2)
            perm_diff = macro_f1(y_true, y_p1_perm) - macro_f1(y_true, y_p2_perm)
            if perm_diff >= obs_diff:
                count += 1
        return (count + 1) / (n_perms + 1)
        
    p_rf = run_permutation(y_test, y_pred_e06, y_pred_rf)
    p_lrs = run_permutation(y_test, y_pred_e06, y_pred_lr_struct)
    p_lrt = run_permutation(y_test, y_pred_e06, y_pred_lr_text)
    
    p_values = [p_rf, p_lrs, p_lrt]
    p_adjusted = holm_bonferroni(p_values)
    
    results = [
        {
            "Comparison": "E06 vs Random Forest Structural",
            "E06 Macro F1": f1_e06,
            "Baseline Macro F1": f1_rf,
            "Observed Delta": f1_e06 - f1_rf,
            "95% CI Low": ci_l_rf,
            "95% CI High": ci_u_rf,
            "Raw p": p_values[0],
            "Adjusted p": p_adjusted[0]
        },
        {
            "Comparison": "E06 vs Structural Logistic Regression",
            "E06 Macro F1": f1_e06,
            "Baseline Macro F1": f1_lr_struct,
            "Observed Delta": f1_e06 - f1_lr_struct,
            "95% CI Low": ci_l_lrs,
            "95% CI High": ci_u_lrs,
            "Raw p": p_values[1],
            "Adjusted p": p_adjusted[1]
        },
        {
            "Comparison": "E06 vs Text Logistic Regression",
            "E06 Macro F1": f1_e06,
            "Baseline Macro F1": f1_lr_text,
            "Observed Delta": f1_e06 - f1_lr_text,
            "95% CI Low": ci_l_lrt,
            "95% CI High": ci_u_lrt,
            "Raw p": p_values[2],
            "Adjusted p": p_adjusted[2]
        }
    ]
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "statistical_results.csv", index=False)
    with open(out_dir / "statistical_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    def interp(row):
        if row["95% CI Low"] > 0 and row["Adjusted p"] < 0.05:
            return "The observed advantage of E06 is statistically supported under the paired test used."
        else:
            return "The observed advantage was not statistically supported."
            
    report = f"""# Statistical Validation Report

### Objective
Determine whether the observed Macro F1 advantage of the E06 Hybrid model over three competing baselines on the repository-disjoint test set is statistically supported.

### Data and Paired Evaluation Design
All models were evaluated on the exact same original repository-disjoint held-out TEST set. The predictions for each sample were paired across models to properly account for sample variance.

### Statistical Methods
**Primary Metric:** Macro F1 difference.

### Bootstrap Methodology
We used a paired bootstrap over the test sample indices. For each iteration (N=10,000), we sampled test indices with replacement, retrieved the paired predictions, and recalculated the Macro F1 difference. 95% Confidence Intervals were derived from the 2.5th and 97.5th percentiles.

### Permutation Methodology
We used a Monte Carlo paired permutation test (N=10,000). Under the null hypothesis that there is no systematic difference between two models, their predictions for any given test sample are exchangeable. For each permutation, we swapped the paired predictions with a 50% probability and recalculated the Macro F1 difference. The one-sided p-value represents the proportion of permutations yielding a difference greater than or equal to the observed difference.

### Multiple-Comparison Correction
Since three comparisons were performed simultaneously against E06, the raw permutation p-values were corrected using the Holm-Bonferroni method to strictly control the Family-Wise Error Rate (FWER) at alpha=0.05.

### Results

| Comparison | E06 Macro F1 | Baseline Macro F1 | Observed Delta | 95% CI Low | 95% CI High | Raw p | Adjusted p |
| ---------- | ------------ | ----------------- | -------------- | ---------- | ----------- | ----- | ---------- |
| E06 vs RF Struct | {f1_e06:.4f} | {f1_rf:.4f} | {f1_e06 - f1_rf:.4f} | {ci_l_rf:.4f} | {ci_u_rf:.4f} | {p_values[0]:.4e} | {p_adjusted[0]:.4e} |
| E06 vs LR Struct | {f1_e06:.4f} | {f1_lr_struct:.4f} | {f1_e06 - f1_lr_struct:.4f} | {ci_l_lrs:.4f} | {ci_u_lrs:.4f} | {p_values[1]:.4e} | {p_adjusted[1]:.4e} |
| E06 vs LR Text | {f1_e06:.4f} | {f1_lr_text:.4f} | {f1_e06 - f1_lr_text:.4f} | {ci_l_lrt:.4f} | {ci_u_lrt:.4f} | {p_values[2]:.4e} | {p_adjusted[2]:.4e} |

### Effect Sizes
* **E06 vs Random Forest Structural:** Absolute +{f1_e06 - f1_rf:.4f}, Relative improvement: {((f1_e06 - f1_rf) / f1_rf * 100):.2f}%
* **E06 vs Structural Logistic Regression:** Absolute +{f1_e06 - f1_lr_struct:.4f}, Relative improvement: {((f1_e06 - f1_lr_struct) / f1_lr_struct * 100):.2f}%
* **E06 vs Text Logistic Regression:** Absolute +{f1_e06 - f1_lr_text:.4f}, Relative improvement: {((f1_e06 - f1_lr_text) / f1_lr_text * 100):.2f}%

### Interpretation
* **E06 vs Random Forest Structural:** {interp(results[0])}
* **E06 vs Structural Logistic Regression:** {interp(results[1])}
* **E06 vs Text Logistic Regression:** {interp(results[2])}

### Limitations
Explicitly note that this analysis compares performance solely on the existing random repository-disjoint test set and does NOT establish temporal generalization. The permutation test for Macro F1 is computationally sound but is an approximation, as the exact distribution of non-linear metrics under swap permutations can be complex.
"""
    with open(out_dir / "statistical_report.md", "w") as f:
        f.write(report)
        
    print("Done!")

if __name__ == "__main__":
    main()
