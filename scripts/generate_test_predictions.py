import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('.')
from src.models.neural_hybrid import NeuralHybrid
from src.models.structural_mlp import StructuralMLP
from transformers import AutoModelForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_preds(out_dir, preds, labels, sids):
    df = pd.DataFrame({'sample_id': sids, 'true_label': labels, 'pred_label': preds})
    df.to_parquet(os.path.join(out_dir, 'test_predictions.parquet'))
    print(f"Saved predictions to {out_dir}")

def generate_classical_preds():
    print("Generating Classical Predictions...")
    df_hybrid = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    df_struct = pd.read_parquet('data/processed/model_ready_structure_v1.parquet')
    df_text = pd.read_parquet('data/processed/model_ready_text_v1.parquet')
    
    test_hybrid = df_hybrid[df_hybrid['split'] == 'test']
    test_struct = df_struct[df_struct['split'] == 'test']
    test_text = df_text[df_text['split'] == 'test']
    
    sids = test_hybrid['sample_id'].values
    y_test = test_hybrid['final_label'].values
    
    # E02 TF-IDF SVM
    if os.path.exists('experiments/e02_tfidf_linear_svm/model.joblib'):
        model_e02 = joblib.load('experiments/e02_tfidf_linear_svm/model.joblib')
        X_test = model_e02['vectorizer'].transform(test_text['normalized_workflow_text'])
        preds = model_e02['model'].predict(X_test)
        save_preds('experiments/e02_tfidf_linear_svm', preds, y_test, sids)
        
    # E05 Structural HGB
    if os.path.exists('experiments/e05_structure_gbdt/model.joblib'):
        from src.models.structure_baselines import StructureBaseline
        model_e05 = StructureBaseline.load('experiments/e05_structure_gbdt/model.joblib')
        
        with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
            import json
            manifest = json.load(f)
        diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
        feature_cols = [c for c in test_struct.columns if c not in ["sample_id", "final_label", "split"] + diagnostic_features]
        
        X_test_struct = test_struct[feature_cols]
        preds = model_e05.predict(X_test_struct)
        os.makedirs('experiments/e05_structure_gbdt', exist_ok=True)
        save_preds('experiments/e05_structure_gbdt', preds, y_test, sids)
        
    # E06 Hybrid LogReg
    if os.path.exists('experiments/e06_hybrid_logreg/model.joblib'):
        from src.models.hybrid_classical import HybridBaseline
        model_e06 = HybridBaseline.load('experiments/e06_hybrid_logreg/model.joblib')
        
        # We need text and struct for E06
        df_hybrid = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
        test_hybrid = df_hybrid[df_hybrid['split'] == 'test']
        text_col = "normalized_workflow_text"
        
        with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
            import json
            manifest = json.load(f)
        diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
        struct_cols = [c for c in df_hybrid.columns if c not in ["sample_id", "final_label", "split", text_col] + diagnostic_features]
        
        X_test_text_e06 = test_hybrid[text_col]
        X_test_struct_e06 = test_hybrid[struct_cols]
        preds = model_e06.predict(X_test_text_e06, X_test_struct_e06)
        save_preds('experiments/e06_hybrid_logreg', preds, y_test, sids)

class TextDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            'sample_id': item['sample_id'],
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'label': torch.tensor(item['labels'] if 'labels' in item else item['final_label'], dtype=torch.long)
        }

class HybridDataset(Dataset):
    def __init__(self, ds, df_struct, scaler, features):
        self.ds = ds
        df_scaled = scaler.transform(df_struct[features])
        self.struct_map = {sid: feat for sid, feat in zip(df_struct['sample_id'], df_scaled)}
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        struct = self.struct_map[item['sample_id']]
        return {
            'sample_id': item['sample_id'],
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'struct': torch.tensor(struct, dtype=torch.float32),
            'label': torch.tensor(item['labels'] if 'labels' in item else item['final_label'], dtype=torch.long)
        }
        
def generate_neural_preds():
    print("Generating Neural Predictions...")
    ds_text = load_from_disk('data/intermediate/codebert_tokenized/512')
    test_ds = ds_text.filter(lambda x: x['split'] == 'test')
    
    df_struct = pd.read_parquet('data/processed/model_ready_structure_v1.parquet')
    exclude = ['sample_id', 'final_label', 'split', 'permissions_none_count', 'language_count', 'language_diversity', 'declared_language_presence']
    features = [c for c in df_struct.columns if c not in exclude]
    scaler = joblib.load('experiments/e09_structural_mlp/scaler.joblib')
    
    hybrid_test_data = HybridDataset(test_ds, df_struct, scaler, features)
    hybrid_loader = DataLoader(hybrid_test_data, batch_size=16)
    
    text_test_data = TextDataset(test_ds)
    text_loader = DataLoader(text_test_data, batch_size=16)
    
    # E08
    if os.path.exists('experiments/e08_codebert_text/checkpoints'):
        import glob
        ckpts = glob.glob('experiments/e08_codebert_text/checkpoints/checkpoint-*')
        if ckpts:
            best_ckpt = max(ckpts, key=os.path.getmtime)
            model_e08 = AutoModelForSequenceClassification.from_pretrained(best_ckpt, num_labels=3, local_files_only=True).to(device)
            model_e08.eval()
            preds, sids, labels = [], [], []
            with torch.no_grad():
                for b in text_loader:
                    logits = model_e08(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device)).logits
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    labels.extend(b['label'].numpy())
                    sids.extend(b['sample_id'])
            save_preds('experiments/e08_codebert_text', preds, labels, sids)
            del model_e08; torch.cuda.empty_cache()
            
    # E09
    if os.path.exists('experiments/e09_structural_mlp/model.pt'):
        model_e09 = StructuralMLP(input_dim=80).to(device)
        model_e09.load_state_dict(torch.load('experiments/e09_structural_mlp/model.pt'))
        model_e09.eval()
        preds, sids, labels = [], [], []
        with torch.no_grad():
            for b in hybrid_loader:
                logits = model_e09(b['struct'].to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(b['label'].numpy())
                sids.extend(b['sample_id'])
        save_preds('experiments/e09_structural_mlp', preds, labels, sids)
        del model_e09; torch.cuda.empty_cache()

    # E10
    if os.path.exists('experiments/e10_neural_hybrid/model.pt'):
        model_e10 = NeuralHybrid(freeze_codebert=True).to(device)
        model_e10.load_state_dict(torch.load('experiments/e10_neural_hybrid/model.pt'))
        model_e10.eval()
        preds, sids, labels = [], [], []
        with torch.no_grad():
            for b in hybrid_loader:
                with torch.amp.autocast('cuda'):
                    logits = model_e10(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device), structural_features=b['struct'].to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(b['label'].numpy())
                sids.extend(b['sample_id'])
        save_preds('experiments/e10_neural_hybrid', preds, labels, sids)
        del model_e10; torch.cuda.empty_cache()

    # E11
    if os.path.exists('experiments/e11_neural_hybrid_finetuned/model.pt'):
        model_e11 = NeuralHybrid(freeze_codebert=False, freeze_codebert_embeddings=True).to(device)
        model_e11.load_state_dict(torch.load('experiments/e11_neural_hybrid_finetuned/model.pt'))
        model_e11.eval()
        preds, sids, labels = [], [], []
        with torch.no_grad():
            for b in hybrid_loader:
                with torch.amp.autocast('cuda'):
                    logits = model_e11(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device), structural_features=b['struct'].to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(b['label'].numpy())
                sids.extend(b['sample_id'])
        save_preds('experiments/e11_neural_hybrid_finetuned', preds, labels, sids)
        del model_e11; torch.cuda.empty_cache()

if __name__ == '__main__':
    generate_classical_preds()
    generate_neural_preds()
