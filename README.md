# RunSure — CI/CD Workflow Execution-Risk Prediction

### What it predicts
RunSure predicts the execution-risk class of a GitHub Actions YAML workflow: **LOW**, **MEDIUM**, or **HIGH**. The risk class is based on the observed historical execution failure rate of that specific workflow configuration.

### Label definition
The empirical failure rate is calculated as:
\ailure_rate = (failure + timed_out) / (success + failure + timed_out)\

**Thresholds:**
* LOW (T1): <= 0.02535
* MEDIUM: > 0.02535 and <= 0.21053
* HIGH (T2): > 0.21053

### Dataset
* 12,944 workflow versions
* 3,924 unique GitHub repositories

### Best validated models
* **E06 Hybrid Logistic Regression:** 0.4972 Macro F1
* **N2 Hybrid XGBoost:** 0.4974 Macro F1

### Current production model
**E06** (Deployed in Streamlit)

### Research candidate
**N2 Hybrid XGBoost** (Preserved in \experiments/final_candidate/\)

### Temporal limitation
Temporal generalization is **not established**. Because all repositories were consumed in the original randomized partition, a clean temporal split could not be mathematically constructed without severe data leakage. 

### Important invalid experiment
An earlier reported 0.6459 temporal result was audited and found **INVALID** due to repository and workflow leakage (68% of test repos were in the training set). It has been excluded from all claims.

### Limitations
* **Moderate performance:** The classification ceiling remains around ~0.50 Macro F1.
* **Medium-class difficulty:** The non-linear boundaries separating MEDIUM from LOW/HIGH are intrinsically difficult to resolve without deeper architectural hierarchies.
* **Temporal generalization unresolved:** Generalization to future, unseen CI/CD APIs is unknown.
* **Labels:** Based on historical execution outcomes, which conflate semantic CI risk with infrastructure flakiness.
* **Transformer experiment:** Heavily constrained by hardware; full hierarchical sequence fine-tuning is pending GPU availability.
* **External holdout:** A fresh external holdout dataset has not yet been collected.
