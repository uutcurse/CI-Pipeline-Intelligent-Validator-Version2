import os
import time
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from huggingface_hub import snapshot_download

model_id = "microsoft/codebert-base"
out_json = "reports/overnight_status.json"
status_data = {}

print("=== STAGE 1: REAL CODEBERT DOWNLOAD ===")
start_time = time.time()
deadline = start_time + 6 * 3600
download_success = False
retry_count = 0

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

while time.time() < deadline:
    try:
        print(f"Download attempt {retry_count + 1}...")
        path = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.bin", "*.json", "*.txt"],
            resume_download=True,
            local_files_only=False
        )
        download_success = True
        status_data["download"] = {
            "status": "SUCCESS",
            "duration": time.time() - start_time,
            "path": path,
            "retries": retry_count
        }
        break
    except Exception as e:
        print(f"Error: {e}. Retrying in 60s...")
        retry_count += 1
        time.sleep(60)

if not download_success:
    print("BLOCKED - REAL CHECKPOINT UNAVAILABLE")
    status_data["download"] = {"status": "BLOCKED"}
    with open(out_json, "w") as f:
        json.dump(status_data, f)
    exit(1)

print("\n=== STAGE 2: REAL WEIGHT VERIFICATION ===")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
fresh_config = AutoConfig.from_pretrained(model_id, num_labels=3)
fresh_model = AutoModelForSequenceClassification.from_config(fresh_config)

param_count = sum(p.numel() for p in model.parameters())
is_equal = torch.allclose(
    model.roberta.encoder.layer[0].attention.self.query.weight,
    fresh_model.roberta.encoder.layer[0].attention.self.query.weight
)
arch = model.config.architectures[0]

verification_status = "SUCCESS" if (not is_equal and arch == "RobertaModel") else "FAILED"
status_data["verification"] = {
    "status": verification_status,
    "parameter_count": param_count,
    "provenance": "REAL" if not is_equal else "DUMMY",
    "architecture": arch
}

manifest = {
    "source_repository": model_id,
    "local_cache_path": status_data["download"]["path"],
    "parameter_count": param_count,
    "verification_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "verification_status": verification_status
}
with open("data/manifests/codebert_checkpoint_manifest_v1.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\n=== STAGE 3: TOKENIZATION AUDIT ===")
df = pd.read_parquet("data/processed/model_ready_text_v1.parquet")
texts = df["normalized_workflow_text"].tolist()

# Tokenize all
tokens = tokenizer(texts, truncation=False)["input_ids"]
lengths = np.array([len(t) for t in tokens])

audit = {}
for ml in [128, 256, 384, 512]:
    trunc = np.sum(lengths > ml)
    audit[ml] = {
        "truncated": int(trunc),
        "percentage": float(trunc / len(lengths) * 100),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(np.max(lengths))
    }
status_data["audit"] = audit

print("\n=== STAGE 4: TOKENIZED DATA CACHE ===")
ds = Dataset.from_pandas(df)
cache_info = {}
for ml in [256, 384, 512]:
    def tok(batch):
        return tokenizer(batch["normalized_workflow_text"], padding="max_length", truncation=True, max_length=ml)
    
    tok_ds = ds.map(tok, batched=True, batch_size=1000)
    tok_ds = tok_ds.remove_columns([c for c in ds.column_names if c not in ["sample_id", "input_ids", "attention_mask", "final_label", "split"]])
    
    out_path = f"data/intermediate/codebert_tokenized/{ml}"
    tok_ds.save_to_disk(out_path)
    
    cache_info[ml] = {
        "path": out_path,
        "rows": len(tok_ds)
    }
status_data["caches"] = cache_info

print("\n=== STAGE 5: GPU SMOKE / THROUGHPUT BENCHMARK ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

benchmarks = []
for ml in [256, 384, 512]:
    for bs in [1, 2]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            input_ids = torch.randint(0, tokenizer.vocab_size, (bs, ml)).to(device)
            attention_mask = torch.ones((bs, ml)).to(device)
            labels = torch.randint(0, 3, (bs,)).to(device)
            
            # warm up
            model(input_ids, attention_mask=attention_mask, labels=labels)
            
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            torch.cuda.synchronize()
            t1 = time.time()
            
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.time()
            
            fwd_time = t1 - t0
            bwd_time = t2 - t1
            sps = bs / (fwd_time + bwd_time)
            
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_res = torch.cuda.max_memory_reserved() / (1024**2)
            
            benchmarks.append({
                "ml": ml, "bs": bs, "fwd": fwd_time, "bwd": bwd_time,
                "sps": sps, "vram_alloc": peak_alloc, "vram_res": peak_res, "status": "OK"
            })
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                benchmarks.append({"ml": ml, "bs": bs, "status": "OOM"})
                torch.cuda.empty_cache()
            else:
                raise e
status_data["benchmarks"] = benchmarks

with open(out_json, "w") as f:
    json.dump(status_data, f, indent=2)

print("PIPELINE COMPLETE.")
