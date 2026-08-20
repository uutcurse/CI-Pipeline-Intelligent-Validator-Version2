import pytest
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk

def test_codebert_provenance_and_smoke():
    model_name = "microsoft/codebert-base"
    
    # Check no dummy checkpoint
    if os.path.exists(model_name) and os.path.isdir(model_name):
        raise ValueError("Dummy checkpoint detected!")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # 1. Provenance
    assert model.config.architectures == ["RobertaModel"], "Architecture mismatch!"
    
    model.to(device)
    
    # Mock data
    texts = ["name: CI\non: push"] * 2
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 4. 3-class output & 5. CUDA forward
    outputs = model(**inputs)
    assert outputs.logits.shape == (2, 3)
    
    # 6. CUDA backward
    labels = torch.tensor([0, 1]).to(device)
    loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
    loss.backward()
    assert model.roberta.embeddings.word_embeddings.weight.grad is not None

def test_tokenized_dataset_schema():
    ds = load_from_disk("data/intermediate/codebert_tokenized/256")
    
    # 7. tokenized dataset schema
    expected_cols = {"sample_id", "input_ids", "attention_mask", "final_label", "split"}
    assert set(ds.column_names) == expected_cols
    
    # 9. no forbidden fields (ensured by above)
    
    # 8. train/validation/test separation
    splits = set(ds["split"])
    assert splits == {"train", "validation", "test"}
