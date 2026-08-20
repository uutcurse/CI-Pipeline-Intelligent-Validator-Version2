import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import json
import numpy as np
from src.inference.workflow_preprocessor import WorkflowPreprocessor

preds = pd.read_parquet('experiments/e06_hybrid_logreg/final_test_predictions_v1.parquet').head(1)
workflows = pd.read_parquet('data/intermediate/workflow_configurations_v1.parquet')

pp = WorkflowPreprocessor()

for _, row in preds.iterrows():
    idx = int(row['sample_id'].split('_')[1])
    pw = json.loads(workflows.iloc[idx]['parsed_workflow_json'])
    
    for k in ['job_count', 'step_count', 'actions_used', 'triggers', 'languages', 'services', 'runners']:
        pw.pop(k, None)
        
    def unstandardize(obj):
        if not isinstance(obj, dict): return
        repls = {'runs_on': 'runs-on', 'timeout_minutes': 'timeout-minutes', 
                 'continue_on_error': 'continue-on-error', 'working_directory': 'working-directory'}
        keys = list(obj.keys())
        for k in keys:
            v = obj[k]
            if k in repls:
                obj[repls[k]] = v
                del obj[k]
                k = repls[k]
            if isinstance(v, dict): unstandardize(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, dict): unstandardize(x)
                    
    unstandardize(pw)
    import yaml
    yaml_in = yaml.dump(pw, sort_keys=False)
    
    res = pp.predict(yaml_in)
    pred = res['prediction']
    
    true_label = row['predicted_label']
    p0 = row['probability_class_0']
    p1 = row['probability_class_1']
    p2 = row['probability_class_2']
    
    print("PREDICTED:")
    print(pred)
    print("TRUE:")
    print({"label": true_label, "LOW": p0, "MEDIUM": p1, "HIGH": p2})
