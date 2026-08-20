import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import json
import numpy as np
from src.preprocessing.normalize_workflow import normalize_workflow
from src.inference.workflow_preprocessor import WorkflowPreprocessor

preds = pd.read_parquet('experiments/e06_hybrid_logreg/final_test_predictions_v1.parquet').head(50)
hyb = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')

df_target = pd.merge(preds, hyb, on='sample_id', how='inner')

workflows = pd.read_parquet('data/intermediate/workflow_configurations_v1.parquet')
workflows['norm_text'] = workflows['parsed_workflow_json'].apply(lambda x: normalize_workflow(json.loads(x)))

df = pd.merge(df_target, workflows, left_on='normalized_workflow_text', right_on='norm_text', how='inner').head(1)

pp = WorkflowPreprocessor()

for _, row in df.iterrows():
    pw = json.loads(row['parsed_workflow_json'])
    
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
    
    print("PROD:", pred['probabilities'])
    print("TRUE:", p0, p1, p2)
    
    # Also compare features
    prod_features = pp.extract_structural_features(pp.parse_yaml(yaml_in))
    with open('data/manifests/final_model_input_contract_v1.json', 'r') as f:
        features = json.load(f)['inputs']['structure']['features']
    true_features = {f: float(row[f]) for f in features}
    
    mismatches = []
    for k in features:
        if true_features[k] != prod_features[k]:
            mismatches.append(f"{k}: true={true_features[k]}, prod={prod_features[k]}")
            
    if mismatches:
        for m in mismatches:
            print(m)
    else:
        print("Features perfectly match.")
