import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.workflow_preprocessor import WorkflowPreprocessor

with open('tests/fixtures/workflows/gha_fixture_0.yml', 'r') as f:
    yaml_in = f.read()
    
preprocessor = WorkflowPreprocessor()

def benchmark(n):
    start_total = time.time()
    
    t_parse = 0
    t_norm = 0
    t_feat = 0
    t_inf = 0
    
    for _ in range(n):
        s0 = time.time()
        pw = preprocessor.parse_yaml(yaml_in)
        s1 = time.time()
        t_parse += (s1 - s0)
        
        s2 = time.time()
        text = preprocessor.generate_normalized_text(pw)
        s3 = time.time()
        t_norm += (s3 - s2)
        
        s4 = time.time()
        struct = preprocessor.extract_structural_features(pw)
        s5 = time.time()
        t_feat += (s5 - s4)
        
        s6 = time.time()
        # Force inference to load service once if not loaded
        if not preprocessor.service:
            preprocessor.predict(yaml_in)
            
        model_input = {"normalized_workflow_text": text, "structural_features": struct}
        preprocessor.service.predict(model_input["normalized_workflow_text"], model_input["structural_features"])
        s7 = time.time()
        t_inf += (s7 - s6)
        
    total = time.time() - start_total
    print(f"\n--- Benchmark N={n} ---")
    print(f"Parse: {t_parse:.4f}s")
    print(f"Norm:  {t_norm:.4f}s")
    print(f"Feat:  {t_feat:.4f}s")
    print(f"Infer: {t_inf:.4f}s")
    print(f"Total: {total:.4f}s")

# Warmup
preprocessor.predict(yaml_in)

benchmark(1)
benchmark(10)
benchmark(100)
