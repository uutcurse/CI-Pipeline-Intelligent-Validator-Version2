# Statistical Validation Report

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
| E06 vs RF Struct | 0.4972 | 0.4818 | 0.0154 | -0.0075 | 0.0377 | 9.4691e-02 | 9.4691e-02 |
| E06 vs LR Struct | 0.4972 | 0.4738 | 0.0234 | 0.0014 | 0.0449 | 1.8498e-02 | 3.6996e-02 |
| E06 vs LR Text | 0.4972 | 0.4723 | 0.0248 | 0.0110 | 0.0385 | 3.9996e-04 | 1.1999e-03 |

### Effect Sizes
* **E06 vs Random Forest Structural:** Absolute +0.0154, Relative improvement: 3.19%
* **E06 vs Structural Logistic Regression:** Absolute +0.0234, Relative improvement: 4.94%
* **E06 vs Text Logistic Regression:** Absolute +0.0248, Relative improvement: 5.26%

### Interpretation
* **E06 vs Random Forest Structural:** The observed advantage was not statistically supported.
* **E06 vs Structural Logistic Regression:** The observed advantage of E06 is statistically supported under the paired test used.
* **E06 vs Text Logistic Regression:** The observed advantage of E06 is statistically supported under the paired test used.

### Limitations
Explicitly note that this analysis compares performance solely on the existing random repository-disjoint test set and does NOT establish temporal generalization. The permutation test for Macro F1 is computationally sound but is an approximation, as the exact distribution of non-linear metrics under swap permutations can be complex.
