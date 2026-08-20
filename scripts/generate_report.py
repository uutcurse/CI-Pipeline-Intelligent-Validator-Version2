import json
import yaml

with open("reports/overnight_status.json", "r") as f:
    d = json.load(f)

if d.get("download", {}).get("status") == "BLOCKED":
    print("BLOCKED - REAL CHECKPOINT UNAVAILABLE")
    exit(1)

# Pick recommendation
benchmarks = d["benchmarks"]
valid = [b for b in benchmarks if b["status"] == "OK"]

rec_ml = 256
rec_bs = 1
for ml in [512, 384, 256]:
    ml_valid = [b for b in valid if b["ml"] == ml]
    if ml_valid:
        rec_ml = ml
        rec_bs = max(b["bs"] for b in ml_valid)
        break

sps = next(b["sps"] for b in valid if b["ml"] == rec_ml and b["bs"] == rec_bs)

steps_per_epoch = 8954 / rec_bs
time_per_epoch = 8954 / sps
total_time = time_per_epoch * 3

config = {
    "model": "microsoft/codebert-base",
    "tokenizer": "microsoft/codebert-base",
    "max_length": rec_ml,
    "batch_size": rec_bs,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 3,
    "fp16": True,
    "seed": 42,
    "evaluation_strategy": "epoch",
    "checkpointing_strategy": "epoch",
    "early_stopping_patience": 3
}

with open("configs/experiments/e08_codebert_text.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

md = f"""# CodeBERT Overnight Readiness Report

### Download
- **Status**: {d["download"]["status"]}
- **Duration**: {d["download"]["duration"]:.2f}s
- **Source**: microsoft/codebert-base
- **Retries**: {d["download"]["retries"]}

### Weight Verification
- **Status**: {d["verification"]["status"]}
- **Parameter Count**: {d["verification"]["parameter_count"]:,}
- **Provenance**: {d["verification"]["provenance"]}

### Tokenization
"""
for ml in [128, 256, 384, 512]:
    a = d["audit"][str(ml)]
    md += f"- **{ml}**: {a['truncated']} truncated ({a['percentage']:.2f}%)\n"

md += f"\n- **Mean**: {a['mean']:.1f}\n- **Median**: {a['median']:.1f}\n- **p90**: {a['p90']:.1f}\n- **p99**: {a['p99']:.1f}\n- **Max**: {a['max']}\n"

md += "\n### Tokenized Caches\n"
for ml, c in d["caches"].items():
    md += f"- **{ml}**: {c['path']} ({c['rows']} rows)\n"

md += "\n### RTX 3060 Benchmark\n"
for b in d["benchmarks"]:
    if b["status"] == "OK":
        md += f"- **{b['ml']} / {b['bs']}**: Fwd: {b['fwd']:.3f}s, Bwd: {b['bwd']:.3f}s, VRAM: {b['vram_alloc']:.1f}MB / {b['vram_res']:.1f}MB, {b['sps']:.1f} samples/sec\n"
    else:
        md += f"- **{b['ml']} / {b['bs']}**: OOM\n"

md += f"""
### Recommended Configuration
- **Max Length**: {rec_ml}
- **Batch Size**: {rec_bs}
- **Gradient Accumulation**: 8
- **FP16**: True

### Estimated Training Time
- **Steps/epoch**: {steps_per_epoch:.0f}
- **Time/epoch**: {time_per_epoch / 60:.1f} minutes
- **3 Epochs**: {total_time / 60:.1f} minutes

### Tests
tests/test_codebert_text.py updated.
"""

with open("reports/codebert_overnight_readiness_v1.md", "w") as f:
    f.write(md)

manifest2 = {
    "recommended_config": config,
    "training_estimate_mins": total_time / 60
}
with open("data/manifests/codebert_readiness_manifest_v1.json", "w") as f:
    json.dump(manifest2, f, indent=2)

print("Report generated.")
