# Baseline Models Comparison

This directory contains the results of the research baseline comparison designed to benchmark the production E06 Hybrid Logistic Regression against simpler alternatives. 

1. **Dataset**: 12,944 workflow versions (model_ready_hybrid_v1.parquet).
2. **Split**: 70/15/15 random stratified, strictly grouped by repository to prevent data leakage.
3. **Models**:
   - B0: Majority class (determined solely on TRAIN).
   - B1: Logistic Regression (Structural only).
   - B2: Logistic Regression (Text only).
   - B3: Random Forest (Structural only).
   - B4: XGBoost (Structural only).
   - B5: Existing E06 Hybrid Logistic Regression (Loaded from deployed artifact).
4. **Feature representations**:
   - Text features were processed using a TfidfVectorizer (fit purely on TRAIN).
   - Structural features were standardized using a StandardScaler (fit purely on TRAIN).
5. **Evaluation metrics**: Macro F1 (primary), Accuracy, Weighted F1, Balanced Accuracy, MCC, Log Loss, ROC-AUC.
6. **Leakage controls**: Rigorous checks assert Train n Val = 0, Train n Test = 0, Val n Test = 0.
7. **Results & Ranking**: See aseline_results.csv for raw metrics, and aseline_report.md for interpretation. Ranking is based strictly on Macro F1.
