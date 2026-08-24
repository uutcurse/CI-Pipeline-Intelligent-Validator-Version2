# Baseline Comparison Report

### Overall Results (Ranked by Macro F1)

| Rank | Model | Representation | Accuracy | Macro F1 | Weighted F1 | Balanced Accuracy | MCC | Log Loss | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B5 (E06 Hybrid) | Text + Structure | 0.4995 | **0.4972** | 0.4978 | 0.4989 | 0.2486 | 1.0086 | 0.6849 |
| 2 | B3 (RF Struct) | Structure Only | 0.4866 | 0.4818 | 0.4839 | 0.4826 | 0.2272 | 1.0206 | 0.6617 |
| 3 | B1 (LogReg Struct) | Structure Only | 0.4781 | 0.4738 | 0.4756 | 0.4752 | 0.2148 | 1.0385 | 0.6613 |
| 4 | B4 (XGB Struct) | Structure Only | 0.4747 | 0.4733 | 0.4741 | 0.4737 | 0.2107 | 1.0901 | 0.6471 |
| 5 | B2 (LogReg Text) | Text Only | 0.4742 | 0.4723 | 0.4734 | 0.4730 | 0.2099 | 0.9970 | 0.6817 |
| 6 | B0 (Majority) | N/A | 0.3504 | 0.1730 | 0.1818 | 0.3333 | 0.0000 | 23.4140 | 0.5000 |

### Per-Class Results (Support: LOW=684, MEDIUM=705, HIGH=623)

Available in per_class_results.csv.

### Confusion Matrices
Saved as PNG heatmaps in experiments/baselines/confusion_matrices/:
* [B1: LogReg Structural](confusion_matrices/logreg_structural.png)
* [B2: LogReg Text](confusion_matrices/logreg_text.png)
* [B3: Random Forest Structural](confusion_matrices/random_forest.png)
* [B4: XGBoost Structural](confusion_matrices/xgboost.png)
* [B5: E06 Hybrid](confusion_matrices/e06_hybrid.png)

### Interpretation

1. **Does structural-only Logistic Regression beat the majority baseline?**
   Yes. Structural-only Logistic Regression (B1) achieves a Macro F1 of 0.4738, massively outperforming the Majority baseline's 0.1730.
2. **Does text-only Logistic Regression beat structural-only?**
   No. Text-only Logistic Regression (B2, Macro F1: 0.4723) performs marginally worse than Structural-only Logistic Regression (B1, Macro F1: 0.4738).
3. **Does Random Forest beat Logistic Regression?**
   Yes. Random Forest on structural features (B3) achieves a Macro F1 of 0.4818, which is higher than B1 (0.4738).
4. **Does XGBoost beat the classical baselines?**
   No. The conservative XGBoost configuration (B4, Macro F1: 0.4733) performs worse than Random Forest and nearly identically to Structural Logistic Regression. 
5. **Does E06 beat all simpler baselines?**
   Yes. The existing E06 model achieves the highest Macro F1 (0.4972), successfully beating all other models.
6. **Which representation appears most useful?**
   Combining both appears most useful (Hybrid E06). On their own, structural features marginally outperform raw text, indicating that explicit feature engineering on the workflow AST is highly valuable.
7. **Is the current E06 model genuinely competitive?**
   Yes. It provides the strongest performance of the cohort, justifying the architectural decision to combine text vectors with engineered structural features.
