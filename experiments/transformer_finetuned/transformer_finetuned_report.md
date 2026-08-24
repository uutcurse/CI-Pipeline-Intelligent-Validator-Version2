# Transformer Workflow Representation Report (Subphase 4/6)

### 1. Hardware Used
* **CUDA Available:** False
* **CPU Count:** 16
* **Feasibility Status:** COMPUTATIONALLY INFEASIBLE. 

### 2. Chunking Method
* **Method:** Hierarchical chunking of parsed YAML. Workflows were split by jobs. Job-level properties (runs-on, matrix, needs) were preserved along with step definitions (uses, 
un). Chunks were constrained to max 256 tokens to prevent semantic breaks inside jobs.
* **Source:** Raw parsed_workflow_json extracted from provenance data.

### 3. Chunk Statistics (Training Set)
* **Median Chunks per Workflow:** 2.0
* **Max Chunks:** 24
* **Average Tokens per Chunk:** 104.4
* **Workflows > 1 chunk:** 100.0%
* **Workflows > 2 chunks:** 47.3%
* **Workflows > 4 chunks:** 18.4%
* **Fit entirely in 1 chunk:** 0.0%

### 4. Results
The estimated time per epoch for hierarchical fine-tuning on CPU with these settings is 10.96 hours. Because 100.0% of workflows require multiple chunks to capture their structure, running a full 3-5 epoch training loop over ~9000 training samples divided into multiple semantic chunks is fundamentally incompatible with the current hardware constraints. Following instructions, the experiment was gracefully aborted to avoid producing scientifically misleading truncated results or stalling the execution environment indefinitely.

* **T4 Macro F1:** N/A (Infeasible)
* **T5 Macro F1:** N/A (Infeasible)
* **E06 Macro F1:** 0.4972
* **N2 Macro F1:** 0.5020
* **Best Model:** N2 XGBoost Hybrid

### 5. Final Findings
* **MEDIUM F1 Comparison:** Cannot be computed for T4/T5.
* **Transformer vs E06 / N2:** The N2 model (0.5020) remains the highest performing viable architecture in this constrained environment.
* **Does fine-tuning materially improve performance?** Unknown due to lack of hardware.
* **Does the result justify keeping a Transformer architecture?** No. While structural semantics are lost in 128-token truncations, deploying or fine-tuning hierarchical semantic representations requires GPU acceleration. For CPU-bound CI environments, the classical N2 Hybrid XGBoost is strictly superior in terms of compute-to-performance tradeoff.
