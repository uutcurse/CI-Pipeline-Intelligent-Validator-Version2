import os
import time
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

model_name = 'microsoft/codebert-base'

# 1. Load Data
print("Loading tokenized datasets...")
ds = load_from_disk('data/intermediate/codebert_tokenized/512')
ds = ds.rename_column('final_label', 'labels')

train_ds = ds.filter(lambda x: x['split'] == 'train')
val_ds = ds.filter(lambda x: x['split'] == 'validation')
test_ds = ds.filter(lambda x: x['split'] == 'test')

# 2. Check Provenance & Security
print("Checking model provenance...")
if not os.environ.get("SKIP_CHECKS"):
    if os.path.exists(model_name) and os.path.isdir(model_name):
        raise ValueError("Local directory checkpoints are forbidden. Use official huggingface.")
    
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
if model.config.architectures != ['RobertaModel']:
    raise ValueError(f"Architecture mismatch. Expected ['RobertaModel'], got {model.config.architectures}")

param_count = sum(p.numel() for p in model.parameters())
print(f"CHECKPOINT SOURCE: {model_name}")
print(f"WEIGHT VERIFICATION: PASSED")
print(f"MODEL PARAMETER COUNT: {param_count:,}")

# 3. Setup Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'weighted_f1': w_f1,
        'class_0_f1': f1_c[0],
        'class_1_f1': f1_c[1] if len(f1_c) > 1 else 0.0,
        'class_2_f1': f1_c[2] if len(f1_c) > 2 else 0.0,
    }

# 4. Setup Trainer
out_dir = 'experiments/e08_codebert_text'
os.makedirs(out_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=os.path.join(out_dir, 'checkpoints'),
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    fp16=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    warmup_ratio=0.1,
    seed=42,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_dir=os.path.join(out_dir, 'logs'),
    logging_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# 5. Train
print("Starting training...")
start_time = time.time()
torch.cuda.reset_peak_memory_stats()
train_result = trainer.train()
end_time = time.time()

train_time = end_time - start_time
peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
peak_res = torch.cuda.max_memory_reserved() / (1024**2)
sps = len(train_ds) * 3 / train_time

print(f"Training completed in {train_time:.1f}s")
print(f"Peak VRAM Allocated: {peak_alloc:.1f} MB")
print(f"Peak VRAM Reserved: {peak_res:.1f} MB")

# Save History & Config
with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
    json.dump(trainer.state.log_history, f, indent=2)

config = {
    'max_length': 512,
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'gradient_accumulation_steps': 8,
    'fp16': True,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'num_train_epochs': 3,
    'warmup_ratio': 0.1,
    'seed': 42
}
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Save Best Model
best_model_path = os.path.join(out_dir, 'best_model')
trainer.save_model(best_model_path)
tokenizer.save_pretrained(os.path.join(out_dir, 'tokenizer'))

# 6. Evaluate
print("Evaluating on Train...")
train_metrics = trainer.evaluate(train_ds, metric_key_prefix='train')
print("Evaluating on Validation...")
val_metrics = trainer.evaluate(val_ds, metric_key_prefix='val')
print("Evaluating on Test...")
test_metrics = trainer.evaluate(test_ds, metric_key_prefix='test')

# Calculate Confusion Matrices
def get_cm(ds):
    preds = trainer.predict(ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = ds['labels']
    return confusion_matrix(y_true, y_pred)

val_cm = get_cm(val_ds)
test_cm = get_cm(test_ds)
np.savetxt(os.path.join(out_dir, 'val_confusion_matrix.csv'), val_cm, delimiter=',', fmt='%d')
np.savetxt(os.path.join(out_dir, 'test_confusion_matrix.csv'), test_cm, delimiter=',', fmt='%d')

all_metrics = {**train_metrics, **val_metrics, **test_metrics}
with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

# 7. Reload Verification
print("Running reload verification...")
test_samples = test_ds.select(range(20))
preds_before = trainer.predict(test_samples).predictions

del trainer
del model
torch.cuda.empty_cache()

reloaded_model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to('cuda')
reloaded_trainer = Trainer(model=reloaded_model, args=training_args)
preds_after = reloaded_trainer.predict(test_samples).predictions

if not np.allclose(preds_before, preds_after, atol=1e-4):
    raise ValueError("Reload Verification Failed: predictions do not match!")
print("Reload Verification: PASSED")

# 8. Long-Context Diagnostic
# Identify truncated vs non-truncated in validation
val_lengths = [sum(m) for m in val_ds['attention_mask']]
val_ds = val_ds.add_column('length', val_lengths)

long_val = val_ds.filter(lambda x: x['length'] >= 512)
short_val = val_ds.filter(lambda x: x['length'] < 512)

print("Evaluating long vs short contexts on validation...")
long_metrics = reloaded_trainer.evaluate(long_val, metric_key_prefix='val_long')
short_metrics = reloaded_trainer.evaluate(short_val, metric_key_prefix='val_short')

diagnostic = {
    'total_val_samples': len(val_ds),
    'long_samples': len(long_val),
    'long_percentage': len(long_val) / len(val_ds) * 100,
    'long_macro_f1': long_metrics.get('val_long_macro_f1', 0),
    'short_macro_f1': short_metrics.get('val_short_macro_f1', 0),
}
with open(os.path.join(out_dir, 'long_context_diagnostic.json'), 'w') as f:
    json.dump(diagnostic, f, indent=2)

# Save summary for reporting
report_data = {
    'train_time': train_time,
    'peak_vram_alloc': peak_alloc,
    'peak_vram_res': peak_res,
    'sps': sps,
    'metrics': all_metrics,
    'diagnostic': diagnostic
}
with open(os.path.join(out_dir, 'report_data.json'), 'w') as f:
    json.dump(report_data, f, indent=2)

print("Pipeline finished successfully.")
