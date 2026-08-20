import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

def test_inference_reload():
    out_dir = Path("experiments/e08_codebert_text")
    if not out_dir.exists():
        print("Model not trained yet.")
        return
        
    print("Loading reloaded model...")
    tokenizer = AutoTokenizer.from_pretrained(out_dir / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(out_dir / "best_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load test data
    df = pd.read_parquet("data/processed/model_ready_text_v1.parquet")
    test_df = df[df["split"] == "test"].head(10)
    texts = test_df["normalized_workflow_text"].tolist()
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    preds = np.argmax(outputs.logits.cpu().numpy(), axis=-1)
    print("Reloaded Predictions:", preds)
    print("SUCCESS: Inference reload verified.")

if __name__ == "__main__":
    test_inference_reload()
