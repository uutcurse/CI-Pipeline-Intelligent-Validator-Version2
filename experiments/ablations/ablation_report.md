# Ablation Study Report

### Overview
This study quantifies the exact contribution of each input representation and structural feature family to the production E06 Hybrid Logistic Regression model.

### Primary Comparison Table

| Experiment | Text | Structure | Macro F1 | Balanced Accuracy | MCC |
| ---------- | ---- | --------- | -------- | ----------------- | --- |
| A1 (Struct Only) | No | Yes | 0.4738 | 0.4752 | 0.2148 |
| A2 (Text Only) | Yes | No | 0.4723 | 0.4730 | 0.2099 |
| A3 (Hybrid E06) | Yes | Yes | 0.4972 | 0.4989 | 0.2486 |

### Improvement Calculations
* **Hybrid - Structural (A3 - A1)**: 0.0234 (Observed relative improvement: 4.94%)
* **Hybrid - Text (A3 - A2)**: 0.0248 (Observed relative improvement: 5.26%)

### Secondary Feature-Group Ablations

**Baseline Macro F1 (A3)**: 0.4972

**A4 (Unigrams only, no bigrams)**:
0.4972 → 0.4828 (Delta = -0.0144)

**Dependency features removed**:
0.4972 → 0.4964
Delta = -0.0007 (Relative change: -0.15%)

**Complexity features removed**:
0.4972 → 0.4903
Delta = -0.0069 (Relative change: -1.39%)

**Action features removed**:
0.4972 → 0.4925
Delta = -0.0047 (Relative change: -0.95%)

**Execution features removed**:
0.4972 → 0.4971
Delta = -0.0001 (Relative change: -0.01%)


### Interpretation

1. **Does structure contain predictive information?**
   Yes. Structural-only Logistic Regression (A1) achieves a Macro F1 of 0.4738, demonstrating predictive utility distinct from raw text.
2. **Does text contain predictive information?**
   Yes. Text-only Logistic Regression (A2) achieves a Macro F1 of 0.4723, successfully extracting signals purely from workflow structure/commands.
3. **Does combining text and structure improve performance?**
   Yes. The A3 hybrid (Macro F1: 0.4972) outperforms both individual representations.
4. **Is the improvement large or small?**
   The observed improvement is moderate (approx +4-5% relative gain over either modality alone). This suggests that while text and structure capture overlapping signals, their combination successfully leverages complementary information.
5. **Which structural feature family appears most useful?**
   Based on the ablation drops, removing Complexity features and Action features resulted in the largest F1 score drops, suggesting they contribute most strongly among the structured sets.
6. **Is the hybrid architecture justified by the observed evidence?**
   Yes. The hybrid architecture empirically maximizes classification performance over any single modality without necessitating an excessively heavy feature space.
