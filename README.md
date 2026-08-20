# CI Workflow Risk Validator

## Project Title
CI Workflow Risk Validator

## Problem Statement
Broken or poorly configured CI/CD workflows waste developer time, compute resources, and delay shipping code. Identifying risky workflow patterns before pushing commits remains difficult because failures are typically only caught during actual CI execution.

## Objective
Build an analytical system that statically parses a GitHub Actions YAML configuration before execution and classifies its expected reliability risk into three discrete tiers (LOW, MEDIUM, HIGH) utilizing historical correlation data.

## Why the problem matters
Preemptive risk identification enables shift-left CI/CD debugging, empowering developers to fix structural anti-patterns before burning cloud compute minutes.

## Input
Raw GitHub Actions workflow configuration strings (.yml, .yaml).

## Output
A predicted risk class (LOW, MEDIUM, or HIGH) alongside three probability percentages evaluating the likelihood across all three classes. *(Note: These probabilities are model correlations, not absolute guarantees of failure or success).*

## System Architecture
A strictly decoupled system:
1. **React Frontend**: Lightweight SPA dashboard.
2. **FastAPI Backend**: Real-time REST endpoints parsing JSON requests.
3. **WorkflowPreprocessor**: Deterministically extracts exactly 80 semantic structural features and text tokens.
4. **Frozen Model (E06)**: TF-IDF + Structural Logistic Regression artifact evaluating inputs in ~4ms.

For a detailed view, see [Architecture](docs/architecture.png) and [Technical Report](docs/technical_report.md).

## Dataset
Derived from ~13k open-source GitHub Actions configurations spanning multiple languages and ecosystem triggers. 

## Label Construction
Calculated from observed repository execution success rates grouped into logical tertiles. (Note: ailure_rate is the target truth, never an input feature).

## Leakage Prevention
Split using strict epo-level isolation to guarantee the model evaluates completely novel structures during validation and testing without memorization overlap.

## Feature Engineering
Extracts complexity bounds, permission structures, 3rd party dependency footprints, and text semantics statically without invoking shell or git states.

## Model Experiments
Evaluated a comprehensive experimental ladder extending from Classical text-only approaches up to Transformer (CodeBERT) semantic hybrids. 

## Final Model
**PRIMARY MODEL**: E06 — TF-IDF + Structural Logistic Regression. 
Chosen for maximum validation performance (Macro-F1 0.4857) and robust generalization boundaries.

## Final Metrics (Frozen Test Set)
- **Macro-F1**: 0.4972
- **Accuracy**: 0.4995

## API
Hosts GET /health, GET /model-info, and POST /predict. Operates entirely stateless without logging raw user payloads.

## Frontend
Built in React + Vite. Features local file parsing, interactive result binding, and strict memory boundaries preventing >1MB uploads.

## Security
Guaranteed safe: No eval, no subprocess shell invocation, strict yaml.safe_load, restricted CORS constraints.

## How to install
Requires Python 3.11 and Node v18+.
\\\ash
# Python dependencies
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements-final.txt

# Node dependencies
cd frontend
npm install
\\\

## How to run backend
\\\ash
python scripts/run_api.py
\\\
*(Hosted at http://127.0.0.1:8080)*

## How to run frontend
In a separate terminal:
\\\ash
cd frontend
npm run dev
\\\
*(VITE_API_BASE_URL defaults to http://127.0.0.1:8080)*

## Example prediction
\\\
LOW: 0.83
MEDIUM: 0.09
HIGH: 0.08
\\\

## Project structure
- data/ - Raw, intermediate, and processed corpus snapshots.
- src/ - Production extraction logic and pipeline architectures.
- configs/ - Project and API YAML configurations.
- experiments/ - Checkpoints and serialized .joblib model branches.
- eports/ - Generated final analysis documents.
- rontend/ - React dashboard codebase.
- 	ests/ - Comprehensive unit and integration coverage.
- docs/ - Model cards, architecture diagrams, and security notes.
- scripts/ - Automated evaluation, training, and demo scripts.

## Testing
100% verified test-suite.
\\\ash
pytest tests/
cd frontend && npm test
\\\

## Reproducibility
The analytical pipeline maintains exact equivalence between research evaluation and production implementation. Verify integrity manually via:
\\\ash
python scripts/reproducibility_check.py
\\\

## Limitations
- Performance is moderate (~0.50 F1) given the extreme complexity of CI environments.
- Model evaluates correlative structures statically, failing to capture dynamic external outages or API rate limits.
- Valid for general-purpose structural configurations, but sensitive to data drift.

## Future work
- Temporal evaluations across drifting repository configurations.
- GNN representations modeling precise job-step DAG logic.

## Research findings
Classical structural integrations cleanly outperformed un-tuned CodeBERT text embeddings within this specific domain bound. Production translations successfully mirrored research logic without a single floating-point deviation across verification sets.
