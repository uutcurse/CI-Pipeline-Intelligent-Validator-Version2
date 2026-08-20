import pandas as pd
from transformers import BertTokenizer, BertTokenizer

def main():
    print("Loading tokenizer microsoft/codebert-base...")
    tokenizer = BertTokenizer.from_pretrained("microsoft/codebert-base")
    
    print("Loading data...")
    df = pd.read_parquet("data/processed/model_ready_text_v1.parquet")
    
    # We only care about the distribution
    texts = df["normalized_workflow_text"].tolist()
    
    print("Tokenizing...")
    # Get length of all texts. We don't need padding or truncation here, just length
    lengths = []
    for t in texts:
        tokens = tokenizer.encode(t, truncation=False)
        lengths.append(len(tokens))
        
    lengths = pd.Series(lengths)
    
    total = len(lengths)
    
    print(f"\n--- Tokenization Audit ---")
    print(f"Total sequences: {total}")
    for cut in [128, 256, 384, 512]:
        exceeds = (lengths > cut).sum()
        pct = (exceeds / total) * 100
        print(f"Exceeding {cut} tokens: {exceeds} ({pct:.2f}%)")
        
if __name__ == "__main__":
    main()
