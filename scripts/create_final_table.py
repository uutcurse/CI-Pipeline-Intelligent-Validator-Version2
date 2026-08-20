import pandas as pd

ablation_data = [
    {"model": "E02", "representation": "Text", "validation_macro_f1": 0.4754, "test_macro_f1": 0.4728, "test_accuracy": 0.4751, "balanced_accuracy": 0.4748},
    {"model": "E05", "representation": "Struct", "validation_macro_f1": 0.4749, "test_macro_f1": 0.4797, "test_accuracy": 0.4821, "balanced_accuracy": 0.4793},
    {"model": "E06 (PRIMARY_MODEL)", "representation": "Text+Struct", "validation_macro_f1": 0.4857, "test_macro_f1": 0.4972, "test_accuracy": 0.4995, "balanced_accuracy": 0.4989},
    {"model": "E08", "representation": "Text", "validation_macro_f1": 0.4577, "test_macro_f1": 0.4728, "test_accuracy": 0.4766, "balanced_accuracy": 0.4751},
    {"model": "E09", "representation": "Struct", "validation_macro_f1": 0.4745, "test_macro_f1": 0.4803, "test_accuracy": 0.4826, "balanced_accuracy": 0.4804},
    {"model": "E10", "representation": "Text+Struct", "validation_macro_f1": 0.4801, "test_macro_f1": 0.4863, "test_accuracy": 0.4861, "balanced_accuracy": 0.4842},
    {"model": "E11", "representation": "Text+Struct", "validation_macro_f1": 0.4721, "test_macro_f1": 0.4809, "test_accuracy": 0.4816, "balanced_accuracy": 0.4801}
]
df = pd.DataFrame(ablation_data)
df.to_csv('reports/final_model_comparison_v1.csv', index=False)
with open('reports/final_model_comparison_v1.md', 'w') as f:
    f.write(df.to_markdown(index=False))
