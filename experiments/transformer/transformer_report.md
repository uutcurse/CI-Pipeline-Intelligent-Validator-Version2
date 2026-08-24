# Transformer Workflow Representation Report

### 1. Model Used
microsoft/codebert-base (Frozen Encoder, INT8 Dynamic Quantization)

### 2. Tokenization Strategy & Long Workflow Handling
* **Strategy:** Controlled truncation at max_length=128 due to strict CPU compute constraints. A sizable fraction of workflows exceed this limit, meaning much of the structural CI definitions at the bottom of the files were lost. Future iterations must implement job-level chunk pooling.

### 3. Model Results (Macro F1)
* **T1 (CodeBERT + LR):** 0.4039
* **T2 (CodeBERT + MLP):** 0.3997
* **T3 (CodeBERT + Structure):** 0.4686
* **E06 Hybrid Logistic Regression:** 0.4972
* **N2 XGBoost Hybrid:** 0.5020

### 4. MEDIUM-Class Comparison
* E06 MEDIUM F1: 0.4480
* N2 MEDIUM F1: 0.4727
* T3 MEDIUM F1: 0.4376

### 5. Multi-seed Results (Best Model: T3 CodeBERT + Structure)
* Mean Macro F1: 0.4632
* Standard Deviation: 0.0194

### 6. Error Overlap (Best Transformer vs E06)
* Transformer fixes E06: 241
* E06 fixes Transformer: 287

### 7. Core Findings
* **Does Transformer add meaningful improvement?** 
  No. Given the severe max_length=128 truncation, the frozen CodeBERT models performed worse than XGBoost on TF-IDF. This strongly indicates that either (a) global workflow structure lost during truncation is essential, or (b) frozen CodeBERT representations (trained on PL source code, not YAML) lack sufficient domain adaptability without full fine-tuning.
* **Does structure remain useful?** 
  Yes, as demonstrated by the delta between T2 (CodeBERT + MLP) and T3 (CodeBERT + Structure + MLP).
* **Is a more complex architecture justified?**
  Only if deployed with full fine-tuning and hierarchical chunking. Frozen CodeBERT on truncated YAML acts as a weak signal compared to simple bag-of-words.
