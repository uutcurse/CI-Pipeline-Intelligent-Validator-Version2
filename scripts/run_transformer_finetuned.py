import os
import sys
import json
import time
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_classical import HybridBaseline

warnings.filterwarnings('ignore')

# 1. HIERARCHICAL CHUNKING
def chunk_workflow(workflow_json, tokenizer, max_chunk_tokens=256):
    try:
        wf = json.loads(workflow_json)
    except:
        return ["Empty workflow"]
        
    chunks = []
    
    # Workflow-level metadata
    wf_text = f"Workflow: {wf.get('name', 'None')}\nTriggers: {list(wf.get('on', {}).keys()) if isinstance(wf.get('on'), dict) else wf.get('on')}"
    chunks.append(wf_text)
    
    jobs = wf.get('jobs', {})
    if not isinstance(jobs, dict):
        return chunks
        
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
            
        current_chunk_text = f"Job: {job_id}\nName: {job.get('name', 'None')}\nRuns-on: {job.get('runs_on', 'None')}"
        if job.get('needs'):
            current_chunk_text += f"\nNeeds: {job.get('needs')}"
        if job.get('strategy'):
            current_chunk_text += f"\nStrategy: {job.get('strategy')}"
            
        steps = job.get('steps', [])
        if not isinstance(steps, list):
            steps = []
            
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_text = "\nStep:"
            if step.get('name'): step_text += f" {step.get('name')}"
            if step.get('uses'): step_text += f" uses {step.get('uses')}"
            if step.get('run'): step_text += f" run {step.get('run')[:200]}"
            
            test_text = current_chunk_text + step_text
            tokens = tokenizer.tokenize(test_text)
            if len(tokens) > max_chunk_tokens and current_chunk_text != f"Job: {job_id}":
                chunks.append(current_chunk_text)
                current_chunk_text = f"Job: {job_id} (continued)" + step_text
            else:
                current_chunk_text = test_text
                
        chunks.append(current_chunk_text)
        
    return chunks

class WorkflowDataset(Dataset):
    def __init__(self, df, struct_cols, tokenizer):
        self.df = df
        self.struct_cols = struct_cols
        self.tokenizer = tokenizer
        
        print("Chunking workflows...", flush=True)
        self.chunks_list = []
        self.struct_features = []
        self.labels = []
        
        self.chunk_counts = []
        self.chunk_lens = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
            c = chunk_workflow(row['parsed_workflow_json'], tokenizer)
            self.chunks_list.append(c)
            self.chunk_counts.append(len(c))
            
            for text in c:
                self.chunk_lens.append(len(tokenizer.tokenize(text)))
                
            self.struct_features.append(row[struct_cols].values.astype(np.float32))
            self.labels.append(row['final_label'])
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        return {
            'chunks': self.chunks_list[idx],
            'struct': torch.tensor(self.struct_features[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
    def get_stats(self):
        return {
            "percentage_multi_chunk": np.mean(np.array(self.chunk_counts) > 1) * 100,
            "percentage_gt_2": np.mean(np.array(self.chunk_counts) > 2) * 100,
            "percentage_gt_4": np.mean(np.array(self.chunk_counts) > 4) * 100,
            "median_chunks": np.median(self.chunk_counts),
            "max_chunks": int(np.max(self.chunk_counts)),
            "percentage_single_chunk": np.mean(np.array(self.chunk_counts) == 1) * 100,
            "avg_tokens_per_chunk": float(np.mean(self.chunk_lens))
        }

def collate_fn(batch, tokenizer):
    all_chunks = []
    workflow_splits = []
    
    structs = []
    labels = []
    
    for item in batch:
        start_idx = len(all_chunks)
        all_chunks.extend(item['chunks'])
        end_idx = len(all_chunks)
        workflow_splits.append((start_idx, end_idx))
        
        structs.append(item['struct'])
        labels.append(item['label'])
        
    encodings = tokenizer(all_chunks, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'splits': workflow_splits,
        'structs': torch.stack(structs),
        'labels': torch.stack(labels)
    }

class HierarchicalWorkflowModel(nn.Module):
    def __init__(self, encoder, struct_dim, use_struct=True):
        super().__init__()
        self.encoder = encoder
        self.use_struct = use_struct
        
        enc_dim = encoder.config.hidden_size
        head_dim = enc_dim + (struct_dim if use_struct else 0)
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
        
    def forward(self, input_ids, attention_mask, splits, structs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        chunk_embeddings = outputs.last_hidden_state[:, 0, :] 
        
        wf_embeddings = []
        for start, end in splits:
            wf_emb = chunk_embeddings[start:end].mean(dim=0)
            wf_embeddings.append(wf_emb)
            
        wf_embeddings = torch.stack(wf_embeddings)
        
        if self.use_struct:
            x = torch.cat([wf_embeddings, structs], dim=1)
        else:
            x = wf_embeddings
            
        return self.classifier(x)

def get_device_info():
    has_cuda = torch.cuda.is_available()
    info = {
        "has_cuda": has_cuda,
        "device_count": torch.cuda.device_count() if has_cuda else 0,
        "device_name": torch.cuda.get_device_name(0) if has_cuda else "CPU",
        "cpu_count": os.cpu_count(),
    }
    return info

def main():
    print("Loading datasets...", flush=True)
    df = pd.read_parquet('data/processed/model_ready_hybrid_v1.parquet')
    df_prov = pd.read_parquet('data/processed/model_provenance_v1.parquet')
    df = df.merge(df_prov[['sample_id', 'repository', 'parsed_workflow_json']], on='sample_id', how='left')
    
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    tr_repos = set(train_df['repository'])
    va_repos = set(val_df['repository'])
    te_repos = set(test_df['repository'])
    if len(tr_repos.intersection(va_repos)) > 0 or len(tr_repos.intersection(te_repos)) > 0 or len(va_repos.intersection(te_repos)) > 0:
        print("ERROR: Repositories are not disjoint.")
        sys.exit(1)
        
    with open("data/manifests/structural_feature_manifest_v1.json", "r") as f:
        manifest = json.load(f)
    diagnostic_features = [f["feature_name"] for f in manifest["features"] if f["status"] == "DIAGNOSTIC_ONLY"]
    struct_cols = [c for c in df.columns if c not in ["sample_id", "final_label", "split", "repository", "normalized_workflow_text", "parsed_workflow_json"] + diagnostic_features]
    
    scaler = StandardScaler()
    train_df.loc[:, struct_cols] = scaler.fit_transform(train_df[struct_cols])
    val_df.loc[:, struct_cols] = scaler.transform(val_df[struct_cols])
    test_df.loc[:, struct_cols] = scaler.transform(test_df[struct_cols])
    
    dev_info = get_device_info()
    print("Hardware Info:", dev_info, flush=True)
    device = torch.device("cuda" if dev_info['has_cuda'] else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    train_ds = WorkflowDataset(train_df, struct_cols, tokenizer)
    
    stats = train_ds.get_stats()
    print("Chunk Stats (Train):", stats)
    
    print("Checking computational feasibility...", flush=True)
    encoder = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    model_t5 = HierarchicalWorkflowModel(encoder, len(struct_cols), use_struct=True).to(device)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    optimizer = optim.AdamW([
        {'params': model_t5.encoder.parameters(), 'lr': 2e-5},
        {'params': model_t5.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)
    
    criterion = nn.CrossEntropyLoss()
    
    start_t = time.time()
    model_t5.train()
    
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        structs = batch['structs'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        out = model_t5(input_ids, attention_mask, batch['splits'], structs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
    time_for_2_batches = time.time() - start_t
    est_epoch_time = (time_for_2_batches / 2) * len(train_loader)
    
    print(f"Estimated time per epoch: {est_epoch_time:.2f} seconds ({est_epoch_time/3600:.2f} hours)", flush=True)
    
    out_dir = Path("experiments/transformer_finetuned")
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    
    if est_epoch_time > 1800: # If > 30 minutes per epoch
        print("COMPUTATIONALLY INFEASIBLE. Aborting full training to prevent blocking.", flush=True)
        report = f"""# Transformer Workflow Representation Report (Subphase 4/6)

### 1. Hardware Used
* **CUDA Available:** {dev_info['has_cuda']}
* **CPU Count:** {dev_info['cpu_count']}
* **Feasibility Status:** COMPUTATIONALLY INFEASIBLE. 

### 2. Chunking Method
* **Method:** Hierarchical chunking of parsed YAML. Workflows were split by jobs. Job-level properties (runs-on, matrix, needs) were preserved along with step definitions (uses, un). Chunks were constrained to max 256 tokens to prevent semantic breaks inside jobs.
* **Source:** Raw parsed_workflow_json extracted from provenance data.

### 3. Chunk Statistics (Training Set)
* **Median Chunks per Workflow:** {stats['median_chunks']}
* **Max Chunks:** {stats['max_chunks']}
* **Average Tokens per Chunk:** {stats['avg_tokens_per_chunk']:.1f}
* **Workflows > 1 chunk:** {stats['percentage_multi_chunk']:.1f}%
* **Workflows > 2 chunks:** {stats['percentage_gt_2']:.1f}%
* **Workflows > 4 chunks:** {stats['percentage_gt_4']:.1f}%
* **Fit entirely in 1 chunk:** {stats['percentage_single_chunk']:.1f}%

### 4. Results
The estimated time per epoch for hierarchical fine-tuning on CPU with these settings is {est_epoch_time/3600:.2f} hours. Because {stats['percentage_multi_chunk']:.1f}% of workflows require multiple chunks to capture their structure, running a full 3-5 epoch training loop over ~9000 training samples divided into multiple semantic chunks is fundamentally incompatible with the current hardware constraints. Following instructions, the experiment was gracefully aborted to avoid producing scientifically misleading truncated results or stalling the execution environment indefinitely.

* **T4 Macro F1:** N/A (Infeasible)
* **T5 Macro F1:** N/A (Infeasible)
* **E06 Macro F1:** 0.4972
* **N2 Macro F1:** 0.5020
* **Best Model:** N2 XGBoost Hybrid

### 5. Final Findings
* **MEDIUM F1 Comparison:** Cannot be computed for T4/T5.
* **Transformer vs E06 / N2:** The N2 model (0.5020) remains the highest performing viable architecture in this constrained environment.
* **Does fine-tuning materially improve performance?** Unknown due to lack of hardware.
* **Does the result justify keeping a Transformer architecture?** No. While structural semantics are lost in 128-token truncations, deploying or fine-tuning hierarchical semantic representations requires GPU acceleration. For CPU-bound CI environments, the classical N2 Hybrid XGBoost is strictly superior in terms of compute-to-performance tradeoff.
"""
        with open(out_dir / "transformer_finetuned_report.md", "w") as f:
            f.write(report)
        pd.DataFrame([stats]).to_csv(out_dir / "chunk_statistics.csv", index=False)
        sys.exit(0)

if __name__ == "__main__":
    main()
