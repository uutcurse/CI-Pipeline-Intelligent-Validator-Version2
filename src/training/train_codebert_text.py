import sys
import json
import yaml
import datetime
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.evaluation.classification_metrics import evaluate_predictions, save_confusion_matrix_plot

def main():
    print("==================================================")
    print("E08: Pretrained Transformer Baseline (CodeBERT)")
    print("==================================================")
    
    out_dir = Path("experiments/e08_codebert_text")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Detected: {gpu_name} ({vram:.2f} GB VRAM)")
    else:
        print("WARNING: No GPU detected! This will be extremely slow.")
        
    start_time = time.time()
    
    config = {
        "experiment_id": "e08_codebert_text",
        "model_name": "microsoft/codebert-base",
        "max_length": 256,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "fp16": True if torch.cuda.is_available() else False,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 3,
        "warmup_ratio": 0.1,
        "seed": 42
    }
    
    import random
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    if config["model_name"] != "microsoft/codebert-base":
        raise ValueError("Only the official microsoft/codebert-base is permitted.")
    if Path(config["model_name"]).exists():
        raise ValueError("Local directory checkpoints are forbidden.")
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=3)
    
    print("Loading dataset...")
    df = pd.read_parquet("data/processed/model_ready_text_v1.parquet")
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    
    # Hugging Face Dataset format
    def to_hf_dataset(d):
        return Dataset.from_pandas(d[["normalized_workflow_text", "final_label"]].rename(columns={"final_label": "label"}))
        
    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)
    test_ds = to_hf_dataset(test_df)
    
    def tokenize_fn(examples):
        return tokenizer(examples["normalized_workflow_text"], truncation=True, max_length=config["max_length"])
        
    print("Tokenizing datasets...")
    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=["normalized_workflow_text"])
    val_tokenized = val_ds.map(tokenize_fn, batched=True, remove_columns=["normalized_workflow_text"])
    test_tokenized = test_ds.map(tokenize_fn, batched=True, remove_columns=["normalized_workflow_text"])
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        m = evaluate_predictions(labels, predictions)
        return {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"]
        }
        
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        fp16=config["fp16"],
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config["seed"],
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("Starting training...")
    train_result = trainer.train()
    
    # Peak memory tracking (rough)
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU Memory Allocated: {peak_mem:.1f} MB")
        
    print("Evaluating on Validation...")
    val_res = trainer.evaluate(val_tokenized)
    print(f"Validation F1: {val_res['eval_macro_f1']:.4f}")
    
    print("Evaluating on Test...")
    test_res = trainer.evaluate(test_tokenized)
    print(f"Test F1: {test_res['eval_macro_f1']:.4f}")
    
    # Re-extract raw predictions for detailed metrics
    test_preds = trainer.predict(test_tokenized)
    y_test_pred = np.argmax(test_preds.predictions, axis=-1)
    
    full_test_metrics = evaluate_predictions(test_df["final_label"], y_test_pred)
    
    save_confusion_matrix_plot(
        test_df["final_label"], y_test_pred, 
        f"reports/figures/e08_codebert_confusion_matrix_test.png",
        title=f"E08 Test CM"
    )
    
    print("Saving best model and tokenizer...")
    best_dir = out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    
    end_time = time.time()
    
    full_config = {
        **config,
        "training_start": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "training_end": datetime.datetime.fromtimestamp(end_time).isoformat(),
        "runtime_seconds": end_time - start_time,
        "peak_gpu_memory_mb": peak_mem,
        "best_validation_metric": val_res['eval_macro_f1'],
        "test_metrics": full_test_metrics
    }
    
    out_dir.joinpath("config.json").write_text(json.dumps(full_config, indent=4))
    out_dir.joinpath("metrics.json").write_text(json.dumps({
        "validation": {"macro_f1": val_res['eval_macro_f1']},
        "test": full_test_metrics
    }, indent=4))
    
    # Log history
    history = trainer.state.log_history
    out_dir.joinpath("training_history.json").write_text(json.dumps(history, indent=4))
    
    print("Done!")

if __name__ == "__main__":
    main()
