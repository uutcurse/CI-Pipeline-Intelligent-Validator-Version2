# Final Validated Results

This table contains ONLY valid final evidence. The contaminated temporal result (0.6459 Macro F1) has been permanently excluded due to repository leakage.

| Model | Representation | Macro F1 | Accuracy | Notes |
|-------|----------------|----------|----------|-------|
| E06 | TF-IDF + Structure | 0.4972 | 0.4969 | Production baseline |
| N2 | TF-IDF/SVD + Structure + XGBoost | 0.4974 | 0.4990 | Selected research candidate |
| RF | Structure | 0.4818 | 0.4819 | Classical baseline |
| LR Structure | Structure | 0.4738 | 0.4735 | Structure-only baseline |
| LR Text | TF-IDF | 0.4723 | 0.4732 | Text-only baseline |

### N2 (Selected Candidate) Metrics
* **MEDIUM F1:** 0.4561
* **Balanced Accuracy:** 0.4986
* **MCC:** 0.2475
* **Uncalibrated Log Loss:** 0.9949
* **Calibrated Log Loss (Isotonic):** 0.9934

### Important Limitations
* **Temporal Generalization:** NOT ESTABLISHED. A clean temporal experiment could not be constructed from the existing dataset because all 3,924 repositories were already consumed by the original random partition.
* **Statistical Claims:** E06 vs Structural Random Forest is NOT statistically supported (p=0.0947). E06 vs Linear baselines IS statistically supported. N2 vs E06 is not statistically confirmed as significant.
