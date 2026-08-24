# E06 Systematic Error Analysis Report

### 1. Objective
Understand WHY the E06 hybrid model makes mistakes to guide future architectural changes, using purely diagnostic analysis on the existing frozen test set.

### 2. Test Dataset
* **Samples:** 2012
* **Repository Disjointness:** Strictly enforced (0 overlap with Train).

### 3. Overall Error Rate
* **Error Rate:** 50.05%
* **Accuracy:** 49.95%

### 4. Confusion Matrix
* **Largest confusion pair:** MEDIUM -> HIGH (Count: 217)
* **Second largest pair:** HIGH -> MEDIUM (Count: 198)

### 5. Per-Class Errors
* **Hardest Class:** MEDIUM (F1: 0.4480)
* **Best Class:** LOW (F1: 0.5831)

### 6. Confidence Analysis
* **Overconfidence:** The confidence distributions (see plots) show that while correct predictions firmly cluster > 0.80, a substantial portion of incorrect predictions also have high confidence (> 0.70). The model occasionally exhibits overconfidence on complex misclassifications.

### 7. Structural Feature Associations
The features with the largest mean differences between correct and incorrect predictions include:
command_token_count, average_command_length, max_command_length, workflow_name_length, continue_on_error_count.
This suggests errors are frequently associated with workflows of differing structural complexity and third-party action usage.

### 8. Error Taxonomy (Heuristic Categories)
* E1: Low/Medium Boundary Ambiguity (Most common)
* E2: Medium/High Boundary Ambiguity
* E3: Complex/Large Workflow Structure (Associated with high job/step counts)

### 9. Hard Cases
* **High Confidence Errors:** 50 highest confidence mistakes saved.
* **Catastrophic Errors (LOW <-> HIGH):** Several cases show high confidence (> 0.75) where a LOW risk is predicted as HIGH or vice versa, often due to misleading text indicators combined with anomalous structural sizes.

### 10. E06 vs Random Forest Error Comparison
* **Both Correct:** 722
* **Both Wrong:** 750
* **Errors E06 Fixed (RF missed):** 283
* **Errors RF Fixed (E06 missed):** 257
* The Hybrid LogReg model fixes 283 samples that RF structurally fails on (leveraging text), but RF fixes 257 cases where E06 struggles, implying non-linear structural interactions hold value.

### 11. Case Studies
10 representative misclassified workflows documented in case_studies.md.

### 12. Main Findings
* **Largest Weakness:** Boundary separation (Medium vs others) and non-linear feature interactions.
* **Hardest Class Boundary:** MEDIUM -> HIGH and HIGH -> MEDIUM
* **Are errors mainly text, structure, or ambiguous?** Highly ambiguous boundaries. E06 fixes many text-related RF errors, but RF captures non-linear structural patterns that E06 misses.
* **Does RF solve errors E06 misses?** Yes, 257 specific errors were correctly classified by RF but missed by E06.

### 13. Implications for the Next Model
**What capability should the next model improve?**
The evidence strongly points to combining **non-linear structural processing** (where RF excels) with **deep textual representations**. Since E06 (linear hybrid) and RF (non-linear structural) have partially disjoint error sets (283 vs 257), a model that can jointly model non-linear interactions between text and structure (e.g. XGBoost on TF-IDF + Structure, or a neural network that can capture deeper non-linear bounds) would likely resolve the persistent ambiguity errors.
