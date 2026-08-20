import pandas as pd
df = pd.read_parquet('data/processed/model_ready_structure_v1.parquet')
print(df['split'].value_counts())
