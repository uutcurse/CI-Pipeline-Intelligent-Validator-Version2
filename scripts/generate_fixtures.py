import pandas as pd
import json
import yaml
import os

df = pd.read_parquet('data/intermediate/workflow_configurations_v1.parquet')
os.makedirs('tests/fixtures/workflows', exist_ok=True)

# Select 10 diverse workflows based on job count, step count, dependencies
selected = []
for idx, row in df.iterrows():
    pw = json.loads(row['parsed_workflow_json'])
    if len(selected) < 15:
        selected.append({'pw': pw, 'sid': f"gha_fixture_{len(selected)}", "original_pw": json.loads(row['parsed_workflow_json'])})
        
for item in selected:
    pw = item['pw']
    
    # Strip preprocessing meta
    for k in ['job_count', 'step_count', 'actions_used', 'triggers', 'languages', 'services', 'runners']:
        pw.pop(k, None)
        
    # Re-hyphenate known keys so it's a valid GitHub action YAML
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
    
    with open(f"tests/fixtures/workflows/{item['sid']}.yml", 'w') as f:
        yaml.dump(pw, f, sort_keys=False)
        
    # Dump the original JSON for ground truth comparison
    with open(f"tests/fixtures/workflows/{item['sid']}_truth.json", 'w') as f:
        json.dump(item['original_pw'], f)
