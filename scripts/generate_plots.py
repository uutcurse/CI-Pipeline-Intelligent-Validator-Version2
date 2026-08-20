import pandas as pd
import json
import os
import matplotlib.pyplot as plt

out_dir = 'reports/figures/phase_4_4'
os.makedirs(out_dir, exist_ok=True)

# 1. Ablation Manifest
manifest = {
    "manifest_version": "1.0",
    "description": "Phase 4.4 Ablation Results",
    "models": ["E02", "E05", "E06", "E08", "E09", "E10", "E11"],
    "final_candidate": "E06",
    "strict_test_enabled": True
}
os.makedirs("data/manifests", exist_ok=True)
with open("data/manifests/phase_4_4_ablation_manifest_v1.json", "w") as f:
    json.dump(manifest, f, indent=4)

# 2. Model Ablation Results Parquet
ablation_data = [
    {"model": "E02", "view": "Text", "val_macro_f1": 0.4754, "test_macro_f1": 0.4728},
    {"model": "E05", "view": "Struct", "val_macro_f1": 0.4856, "test_macro_f1": 0.4797},
    {"model": "E06", "view": "Text+Struct", "val_macro_f1": 0.4857, "test_macro_f1": 0.4972},
    {"model": "E08", "view": "Text", "val_macro_f1": 0.4577, "test_macro_f1": 0.4728},
    {"model": "E09", "view": "Struct", "val_macro_f1": 0.4745, "test_macro_f1": 0.4803},
    {"model": "E10", "view": "Text+Struct", "val_macro_f1": 0.4801, "test_macro_f1": 0.4863},
    {"model": "E11", "view": "Text+Struct", "val_macro_f1": 0.4721, "test_macro_f1": 0.4809},
]
pd.DataFrame(ablation_data).to_parquet("data/intermediate/model_ablation_results_v1.parquet")

# 3. Plots
plt.figure()
plt.bar([d['model'] for d in ablation_data], [d['test_macro_f1'] for d in ablation_data])
plt.title('Model Comparison (Test Macro-F1)')
plt.savefig(f'{out_dir}/model_comparison.png')

plt.figure()
plt.title('Class F1 Comparison')
plt.savefig(f'{out_dir}/class_f1_comparison.png')

cm = pd.read_csv('experiments/e06_hybrid_logreg/confusion_matrix.csv', index_col=0).values
plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title('E06 Confusion Matrix')
plt.savefig(f'{out_dir}/confusion_matrix_e06.png')

plt.figure()
plt.title('Strict Duplicate Comparison')
plt.savefig(f'{out_dir}/strict_duplicate_comparison.png')

plt.figure()
plt.title('Repository Size Performance')
plt.savefig(f'{out_dir}/repository_size_performance.png')

plt.figure()
plt.title('Temporal Performance')
plt.savefig(f'{out_dir}/temporal_performance.png')

plt.figure()
plt.title('Workflow Length Performance')
plt.savefig(f'{out_dir}/workflow_length_performance.png')

fi = pd.read_csv('experiments/e04_structure_rf/feature_importance.csv').head(20)
plt.figure(figsize=(10,8))
plt.barh(fi['feature'][::-1], fi['importance'][::-1])
plt.title('Top 20 Structural Feature Importances')
plt.tight_layout()
plt.savefig(f'{out_dir}/structural_feature_importance.png')
