# Experiment Manifest

| Experiment | Purpose | Dataset | Split | Model | Primary Metric | Result | Status |
|------------|---------|---------|-------|-------|----------------|--------|--------|
| E06 Evaluation | Verify production baseline | Hybrid V1 | Original | Hybrid Logistic Regression | Macro F1 | 0.4972 | VALID |
| Baseline Comparison | Establish classical boundaries | Hybrid V1 | Original | Structural RF, Structural LR, Text LR | Macro F1 | RF=0.4818, LR_Struct=0.4738, LR_Text=0.4723 | VALID |
| Ablation | Verify input contribution | Hybrid V1 | Original | Hybrid Logistic Regression | Macro F1 | - | VALID |
| Statistical Validation | Test significance vs baselines | Hybrid V1 | Original | Hybrid Logistic Regression | p-value | p=0.0947 (vs RF) | VALID |
| Error Analysis | Identify decision boundary weaknesses | Hybrid V1 | Original | Hybrid LR vs Structural RF | Error Count | N/A | DIAGNOSTIC |
| N2 Nonlinear | Test nonlinear tree architectures | Hybrid V1 | Original | Hybrid XGBoost | Macro F1 | 0.5020 | VALID |
| Frozen Transformer | Test semantic representation | Hybrid V1 | Original | CodeBERT (max_length=128) + MLP | Macro F1 | 0.4686 | VALID (Hardware Constrained) |
| Hierarchical Transformer | Overcome 128-token truncation | Raw YAML | Original | Hierarchical CodeBERT | Macro F1 | N/A | INCOMPLETE/INFEASIBLE |
| Temporal Leakage Audit | Verify out-of-time evaluation | Time-based V1 | Temporal | Hybrid Logistic Regression | Data Overlap | 68% leakage | DIAGNOSTIC |
| Temporal Evaluation | Evaluate on future workflows | Time-based V1 | Temporal | Hybrid Logistic Regression | Macro F1 | 0.6459 | INVALID (Leakage) |
| Clean Temporal Split | Construct new valid temporal split | Time-based V1 | Clean Temporal| N/A | Split Size | 0 | INCOMPLETE (No remaining repos) |
| Final Candidate | Select best classical candidate | Hybrid V1 | Original | Hybrid XGBoost (N2) | Macro F1 | 0.4974 | VALID |
