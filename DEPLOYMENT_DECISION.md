# Deployment Decision

**RECOMMENDATION:** KEEP E06 AS PRODUCTION BASELINE

**RESEARCH CANDIDATE:** RETAIN N2 HYBRID XGBOOST FOR FUTURE EVALUATION

### Justification

The extensive evaluation subphases demonstrated that the N2 Hybrid XGBoost model is the strongest model overall. However, its final test evaluation yielded a Macro F1 of 0.4974, compared to the deployed E06 model's Macro F1 of 0.4972.

This constitutes a marginal observed improvement of **+0.0002 Macro F1**. 

While N2 improves confidence calibration (reducing high-confidence errors from 14 to 10), this microscopic performance gain does not outweigh the engineering risks and overhead of ripping out a stable, fully integrated production model (E06) for a new architecture.

**Decision:** E06 will remain live in production. No modifications will be made to pp.py or the Streamlit UI. N2 is officially documented and preserved as the Research Candidate in experiments/final_candidate/ for when a fresh, external holdout dataset is collected to properly re-validate state-of-the-art architectures.
