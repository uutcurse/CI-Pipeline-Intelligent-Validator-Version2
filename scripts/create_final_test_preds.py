import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.inference.predict import E06InferencePipeline
import json
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

with open('data/manifests/final_model_input_contract_v1.json', 'r') as f:
    contract = json.load(f)

df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
test_df = df[df['split'] == 'test']
features = contract['inputs']['structure']['features']

pipeline = E06InferencePipeline()

res_list = []
for idx, row in test_df.iterrows():
    text = row['normalized_workflow_text']
    struct_dict = {f: float(row[f]) for f in features}
    pred = pipeline.predict(text, struct_dict)
    
    res_list.append({
        'sample_id': row['sample_id'],
        'true_label': row['final_label'],
        'predicted_label': pred['predicted_class'],
        'probability_class_0': pred['probabilities'][0],
        'probability_class_1': pred['probabilities'][1],
        'probability_class_2': pred['probabilities'][2]
    })
    
out_df = pd.DataFrame(res_list)
out_df.to_parquet('experiments/e06_hybrid_logreg/final_test_predictions_v1.parquet')
