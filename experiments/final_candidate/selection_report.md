# Final Candidate Selection Report

### 1. Candidate Models
* C1: E06-equivalent Hybrid Logistic Regression
* C2: N2-equivalent Hybrid XGBoost
* C3: Structural Random Forest

### 2. Validation Results
* **C1 (LR) Val Macro F1:** 0.4789
* **C2 (XGB) Val Macro F1:** 0.4830
* **C3 (RF) Val Macro F1:** 0.4713

### 3. Selection Rule
The candidate with the highest validation Macro F1 was selected.

### 4. Selected Candidate
**C2_N2_Hybrid_XGB**

### 5. Final Test Results (Evaluated ONCE)
* **Macro F1:** 0.4974
* **Accuracy:** 0.4990

### 6. Comparison against E06
* **Selected Candidate F1:** 0.4974
* **E06 F1:** 0.4972

### 7. MEDIUM-class Performance
* **Precision:** 0.4673
* **Recall:** 0.4454
* **F1:** 0.4561

### 8. Seed Robustness (Validation)
* **Mean Val Macro F1:** 0.4784
* **Std Val Macro F1:** 0.0055

### 9. Limitations
The final candidate was selected using validation performance; the test set was used only for final evaluation.
