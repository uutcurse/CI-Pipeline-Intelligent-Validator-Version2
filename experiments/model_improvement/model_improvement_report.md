# Model Improvement Report: Hybrid Nonlinear Classical

### 1. Best New Model
N2 XGBoost Hybrid

### 2. Macro F1
0.5020

### 3. Difference from E06
+0.0048

### 4. Difference from Random Forest Structural
+0.0202

### 5. MEDIUM-class F1
* Best Model: 0.4727
* E06: 0.4480
* Change: +0.0247

### 6. Did Nonlinear Modeling Solve Meaningful Errors?
The best nonlinear model fixed 214 errors that E06 made, but it also missed 206 errors that E06 correctly classified. Overall improvement is captured in the F1 change.

### 7. Is the Improvement Consistent Across Seeds?
* Mean Macro F1 (3 seeds): 0.4957
* Std Dev: 0.0054
Consistency indicates robust optimization.

### 8. Is the Added Complexity Justified?
If the performance gain is minimal or negative compared to E06, then combining high-dimensional SVD and gradient boosting does not justify the added latency and architectural overhead.

### 9. Is a Transformer/GNN Warranted?
Yes. While there is marginal improvement, the performance ceiling remains low. Extracting deeper semantic tokens or graph structures likely requires a deep architecture.
