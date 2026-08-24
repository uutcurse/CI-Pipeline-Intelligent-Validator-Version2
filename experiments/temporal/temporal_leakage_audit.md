# Temporal Leakage Audit Report

### Original E06 Split
* **Train repositories:** 2746
* **Validation repositories:** 589
* **Test repositories:** 589

### Temporal Split
* **Train repositories:** 3140
* **Validation repositories:** 198
* **Test repositories:** 205

### Cross-Split Overlaps

| Comparison | Overlap repositories | Percentage |
|---|---:|---:|
| Original E06 Train ∩ Temporal Test | 140 | 68.29% |
| Original E06 Validation ∩ Temporal Test | 23 | 11.22% |
| Original E06 Test ∩ Temporal Test | 42 | 20.49% |

### Workflow-Level Overlap (Original E06 Train vs Temporal Test)
* **Overlapping repositories:** 140
* **Overlapping workflow versions (exact sample matches):** 287
* **Same-workflow-identity matches (same repository + workflow path):** 204

### Analysis
CASE B: Repository overlap detected. The temporal evaluation must be considered potentially contaminated. Do NOT call the 0.6459 result a valid unseen-repository temporal generalization result.

CASE C: Exact workflow versions overlap. The temporal evaluation is strictly invalid due to data leakage.
