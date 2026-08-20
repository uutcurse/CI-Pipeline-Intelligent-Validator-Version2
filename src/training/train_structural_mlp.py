import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from src.models.structural_mlp import StructuralMLP
from src.evaluation.neural_metrics import compute_neural_metrics

out_dir = 'experiments/e09_structural_mlp'
fig_dir = 'reports/figures/e09_structural_mlp'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs('data/intermediate', exist_ok=True)

# 1. Load Data
df = pd.read_parquet('data/processed/model_ready_structure_v1.parquet')

# Features to exclude
exclude = ['sample_id', 'final_label', 'split',
           'permissions_none_count', 'language_count',
           'language_diversity', 'declared_language_presence']
features = [c for c in df.columns if c not in exclude]
assert len(features) == 80, f"Expected 80 features, got {len(features)}"

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'validation']
test_df = df[df['split'] == 'test']

# 2. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_val = scaler.transform(val_df[features])
X_test = scaler.transform(test_df[features])

y_train = train_df['final_label'].values
y_val = val_df['final_label'].values
y_test = test_df['final_label'].values

joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))

# 3. DataLoaders
def make_loader(X, y, batch_size=64, shuffle=False):
    t_X = torch.tensor(X, dtype=torch.float32)
    t_y = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(t_X, t_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

# 4. Model & Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

model = StructuralMLP(input_dim=80).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
criterion = nn.CrossEntropyLoss()

epochs = 50
patience = 7
best_val_f1 = -1
epochs_no_improve = 0

history = {
    'train_loss': [], 'val_loss': [],
    'train_macro_f1': [], 'val_macro_f1': []
}

def evaluate(loader):
    model.eval()
    losses = []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
    
    metrics = compute_neural_metrics(all_labels, all_preds)
    return np.mean(losses), metrics, all_preds, all_labels

print("Starting training...")
start_time = time.time()
torch.cuda.reset_peak_memory_stats()

for epoch in range(epochs):
    model.train()
    train_losses = []
    train_preds, train_labels = [], []
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(y_b.cpu().numpy())
        
    train_metrics = compute_neural_metrics(train_labels, train_preds)
    val_loss, val_metrics, _, _ = evaluate(val_loader)
    
    scheduler.step(val_metrics['macro_f1'])
    
    history['train_loss'].append(np.mean(train_losses))
    history['val_loss'].append(val_loss)
    history['train_macro_f1'].append(train_metrics['macro_f1'])
    history['val_macro_f1'].append(val_metrics['macro_f1'])
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f}")
    
    if val_metrics['macro_f1'] > best_val_f1:
        best_val_f1 = val_metrics['macro_f1']
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

train_time = time.time() - start_time
peak_alloc = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
peak_res = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0

# 5. Evaluate Best Model
model.load_state_dict(torch.load(os.path.join(out_dir, 'model.pt')))

_, final_train_metrics, _, _ = evaluate(train_loader)
_, final_val_metrics, val_preds, val_labels = evaluate(val_loader)
_, final_test_metrics, test_preds, test_labels = evaluate(test_loader)

all_metrics = {
    'train': final_train_metrics,
    'val': final_val_metrics,
    'test': final_test_metrics
}
with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

val_cm = confusion_matrix(val_labels, val_preds)
test_cm = confusion_matrix(test_labels, test_preds)
np.savetxt(os.path.join(out_dir, 'val_confusion_matrix.csv'), val_cm, delimiter=',', fmt='%d')
np.savetxt(os.path.join(out_dir, 'test_confusion_matrix.csv'), test_cm, delimiter=',', fmt='%d')

# Save Config
config = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs': epochs,
    'early_stopping_patience': patience,
    'seed': 42
}
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Save Plots
plt.figure()
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Loss Curve')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'loss_curve.png'))

plt.figure()
plt.plot(history['train_macro_f1'], label='Train')
plt.plot(history['val_macro_f1'], label='Validation')
plt.title('Macro-F1 Curve')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'macro_f1_curve.png'))

# 6. Extract Embeddings
def extract_embeddings(df_split, X_scaled):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        _, embeddings = model(X_t, return_embedding=True)
        embeddings = embeddings.cpu().numpy()
        
    out_df = df_split[['sample_id', 'split', 'final_label']].copy()
    for i in range(32):
        out_df[f'embedding_{i}'] = embeddings[:, i]
    return out_df

emb_train = extract_embeddings(train_df, X_train)
emb_val = extract_embeddings(val_df, X_val)
emb_test = extract_embeddings(test_df, X_test)

emb_all = pd.concat([emb_train, emb_val, emb_test], ignore_index=True)
emb_all.to_parquet('data/intermediate/structural_embeddings_v1.parquet', index=False)

# 7. Print summary for report
print("\n--- RESULTS ---")
print(f"Val Macro-F1: {final_val_metrics['macro_f1']:.4f}")
print(f"Test Macro-F1: {final_test_metrics['macro_f1']:.4f}")
print(f"Test Accuracy: {final_test_metrics['accuracy']:.4f}")
print(f"Peak VRAM Alloc: {peak_alloc:.1f} MB")
print(f"Peak VRAM Res: {peak_res:.1f} MB")
print(f"Train Time: {train_time:.1f} s")
print(f"Samples/sec: {(len(train_df)*len(history['train_loss']))/train_time:.1f}")

print("DONE.")
