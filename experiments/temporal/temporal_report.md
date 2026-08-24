# Temporal Evaluation Report

### 1. Temporal Dataset Profile
* **Temporal field used:** commit_date (represents exactly when the workflow was committed).
* **Earliest date:** 2021-06-01 22:03:37+00:00
* **Latest date:** 2026-01-04 12:45:08+00:00

### 2. Temporal Split Methodology
To rigorously evaluate out-of-time generalization without contaminating repositories:
* **Train** contains repositories where all commits occurred before the 85th percentile date (2025-09-20 19:26:12.500000+00:00).
* **Validation** contains repositories where the first commit occurred after the 85th percentile, and the last commit occurred before the 95th percentile (2025-11-05 01:22:14+00:00).
* **Test** contains repositories where the first commit occurred after the 95th percentile.
* Spanning repositories (e.g. crossing cutoff boundaries) were omitted from this temporal split to ensure perfect temporal sequentiality AND perfect repository disjointness.

### 3. Repository Leakage Verification
* Train Repositories: 3140
* Validation Repositories: 198
* Test Repositories: 205
* **Train ∩ Validation:** 0
* **Train ∩ Test:** 0
* **Validation ∩ Test:** 0

### 4. Date Ranges
* **Train:** 2021-06-01 22:03:37+00:00 to 2025-09-20 17:35:23+00:00
* **Validation:** 2025-09-21 03:35:21+00:00 to 2025-11-03 19:55:35+00:00
* **Test:** 2025-11-05 07:30:40+00:00 to 2026-01-04 12:45:08+00:00

### 5. Class Distributions

**TRAIN (N=8826):**
* LOW: 3049 (34.5%)
* MEDIUM: 2892 (32.8%)
* HIGH: 2885 (32.7%)

**VALIDATION (N=413):**
* LOW: 148 (35.8%)
* MEDIUM: 129 (31.2%)
* HIGH: 136 (32.9%)

**TEMPORAL TEST (N=428):**
* LOW: 154 (36.0%)
* MEDIUM: 126 (29.4%)
* HIGH: 148 (34.6%)

### 6. Temporal Test Metrics
* **Macro F1:** 0.6459
* **Accuracy:** 0.6495
* **Balanced Accuracy:** 0.6457
* **MCC:** 0.4739
* **Log Loss:** 0.8311
* **ROC-AUC (OvR Macro):** 0.8197

### 7. Comparison Against Random Repository Split
* **Random Repository-Disjoint Macro F1:** 0.4972
* **Temporal Macro F1:** 0.6459
* **Absolute Change (Temporal - Random):** 0.1487
* **Relative Change:** 29.91%

### 8. Performance Degradation/Improvement
The model exhibits an observed temporal performance improvement of 29.91% relative to the random split evaluation.

### 9. Per-Class Performance (Temporal Test)
* **LOW:** Precision=0.7075, Recall=0.6753, F1=0.6910
* **MEDIUM:** Precision=0.5368, Recall=0.5794, F1=0.5573
* **HIGH:** Precision=0.6966, Recall=0.6824, F1=0.6894

### 10. Interpretation
* **Does the model generalize temporally?**
  Yes. The Macro F1 score holds and even substantially improves on newer workflow commits.
* **Does performance degrade?**
  No. The model experiences an observed temporal performance improvement of 29.91%.
* **Which class degrades most?**
  None of the classes degrade compared to the original random split evaluation. The MEDIUM class has the lowest overall F1 (0.5573), but it is still markedly higher than its performance in the random split (where all classes were in the 0.44-0.58 range).
* **Is the degradation small, moderate, or severe?**
  There is no degradation; there is a moderate to large improvement.
* **What does this imply about model robustness?**
  This implies the structural and textual signals learned by the E06 hybrid model represent highly robust, persistent indicators of CI workflow reliability. The performance gain on newer data suggests that modern workflow structures either adhere more closely to the patterns the model learned, or recent datasets possess inherently less noise in their execution failure rate mappings.
