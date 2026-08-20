import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from transformers import get_scheduler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from src.models.neural_hybrid import NeuralHybrid
from src.evaluation.neural_metrics import compute_neural_metrics

class HybridDataset(Dataset):
    def __init__(self, ds, df_struct, scaler, features):
        self.ds = ds
        df_scaled = scaler.transform(df_struct[features])
        self.struct_map = {sid: feat for sid, feat in zip(df_struct['sample_id'], df_scaled)}
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        item = self.ds[idx]
        struct = self.struct_map[item['sample_id']]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'struct': torch.tensor(struct, dtype=torch.float32),
            'label': torch.tensor(item['labels'] if 'labels' in item else item['final_label'], dtype=torch.long)
        }

def main():
    out_dir = 'experiments/e11_neural_hybrid_finetuned'
    fig_dir = 'reports/figures/e11_neural_hybrid_finetuned'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df_struct = pd.read_parquet('data/processed/model_ready_structure_v1.parquet')
    ds_text = load_from_disk('data/intermediate/codebert_tokenized/512')

    exclude = ['sample_id', 'final_label', 'split',
               'permissions_none_count', 'language_count',
               'language_diversity', 'declared_language_presence']
    features = [c for c in df_struct.columns if c not in exclude]

    scaler = joblib.load('experiments/e09_structural_mlp/scaler.joblib')

    train_ds = ds_text.filter(lambda x: x['split'] == 'train')
    val_ds = ds_text.filter(lambda x: x['split'] == 'validation')
    test_ds = ds_text.filter(lambda x: x['split'] == 'test')

    train_data = HybridDataset(train_ds, df_struct, scaler, features)
    val_data = HybridDataset(val_ds, df_struct, scaler, features)
    test_data = HybridDataset(test_ds, df_struct, scaler, features)

    batch_size = 2
    grad_acc = 8
    epochs = 5
    patience = 2

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    print("Initializing model...", flush=True)
    model = NeuralHybrid(
        codebert_name='microsoft/codebert-base',
        structural_model_path='experiments/e09_structural_mlp/model.pt',
        freeze_codebert=False,
        freeze_codebert_embeddings=True
    ).to(device)

    print("Setting up discriminative learning rates...", flush=True)
    param_groups = [
        {'params': model.codebert.encoder.parameters(), 'lr': 2e-5},
        {'params': model.structural_mlp.parameters(), 'lr': 1e-4},
        {'params': model.fusion.parameters(), 'lr': 2e-4}
    ]
    if hasattr(model.codebert, 'pooler'):
        param_groups.append({'params': model.codebert.pooler.parameters(), 'lr': 2e-5})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_steps = (len(train_loader) // grad_acc) * epochs
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

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
            for b in loader:
                input_ids = b['input_ids'].to(device)
                attention_mask = b['attention_mask'].to(device)
                struct = b['struct'].to(device)
                label = b['label'].to(device)
                
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask, struct)
                    loss = criterion(logits, label)
                    
                losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                
        metrics = compute_neural_metrics(all_labels, all_preds)
        return np.mean(losses), metrics, all_preds, all_labels

    print("Starting training...", flush=True)
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    scaler_grad = torch.amp.GradScaler()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} start", flush=True)
        epoch_start_time = time.time()
        model.train()
        train_losses = []
        train_preds, train_labels = [], []
        
        optimizer.zero_grad()
        for i, b in enumerate(train_loader):
            input_ids = b['input_ids'].to(device)
            attention_mask = b['attention_mask'].to(device)
            struct = b['struct'].to(device)
            label = b['label'].to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, struct)
                loss = criterion(logits, label) / grad_acc
                
            scaler_grad.scale(loss).backward()
            
            if (i + 1) % grad_acc == 0 or (i + 1) == len(train_loader):
                scaler_grad.step(optimizer)
                scaler_grad.update()
                optimizer.zero_grad()
                scheduler.step()
                
            train_losses.append(loss.item() * grad_acc)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(label.cpu().numpy())
            
        train_metrics = compute_neural_metrics(train_labels, train_preds)
        val_loss, val_metrics, _, _ = evaluate(val_loader)
        
        epoch_time = time.time() - epoch_start_time
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        
        lrs = [g['lr'] for g in optimizer.param_groups]
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Time: {epoch_time:.1f}s | LRs: {lrs}", flush=True)
        
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
            print("  -> Saved new best checkpoint!", flush=True)
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break

    train_time = time.time() - start_time
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    peak_res = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0

    print("Loading best checkpoint for final evaluation...", flush=True)
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
        
    with open(os.path.join(out_dir, 'classification_report.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    val_cm = confusion_matrix(val_labels, val_preds)
    test_cm = confusion_matrix(test_labels, test_preds)
    np.savetxt(os.path.join(out_dir, 'val_confusion_matrix.csv'), val_cm, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(out_dir, 'test_confusion_matrix.csv'), test_cm, delimiter=',', fmt='%d')

    config = {
        'batch_size': batch_size,
        'gradient_accumulation_steps': grad_acc,
        'learning_rate_codebert': 2e-5,
        'learning_rate_struct': 1e-4,
        'learning_rate_fusion': 2e-4,
        'weight_decay': 0.01,
        'epochs': epochs,
        'early_stopping_patience': patience,
        'seed': 42
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

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

    print("\n--- RESULTS ---", flush=True)
    print(f"Val Macro-F1: {final_val_metrics['macro_f1']:.4f}", flush=True)
    print(f"Test Macro-F1: {final_test_metrics['macro_f1']:.4f}", flush=True)
    print(f"Test Accuracy: {final_test_metrics['accuracy']:.4f}", flush=True)
    print(f"Peak VRAM Alloc: {peak_alloc:.1f} MB", flush=True)
    print(f"Peak VRAM Res: {peak_res:.1f} MB", flush=True)
    print(f"Train Time: {train_time:.1f} s", flush=True)

    print("Running reload verification...", flush=True)
    test_samples = [test_data[i] for i in range(20)]
    input_ids_batch = torch.stack([s['input_ids'] for s in test_samples]).to(device)
    attention_mask_batch = torch.stack([s['attention_mask'] for s in test_samples]).to(device)
    struct_batch = torch.stack([s['struct'] for s in test_samples]).to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            preds_before = model(input_ids_batch, attention_mask_batch, struct_batch).cpu().numpy()

    del model
    torch.cuda.empty_cache()

    reloaded_model = NeuralHybrid(
        codebert_name='microsoft/codebert-base',
        freeze_codebert=False,
        freeze_codebert_embeddings=True
    ).to(device)
    reloaded_model.load_state_dict(torch.load(os.path.join(out_dir, 'model.pt')))
    reloaded_model.eval()

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            preds_after = reloaded_model(input_ids_batch, attention_mask_batch, struct_batch).cpu().numpy()

    if not np.allclose(preds_before, preds_after, atol=1e-4):
        raise ValueError("Reload Verification Failed: predictions do not match!")
    print("Reload Verification: PASSED", flush=True)
    print("DONE.", flush=True)

if __name__ == '__main__':
    main()
