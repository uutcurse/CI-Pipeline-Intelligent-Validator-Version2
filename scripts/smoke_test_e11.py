import torch
import sys
sys.path.append('.')
from src.models.neural_hybrid import NeuralHybrid

device = torch.device('cuda')
model = NeuralHybrid(codebert_name='microsoft/codebert-base', freeze_codebert=False, freeze_codebert_embeddings=True).to(device)

param_groups = [
    {'params': model.codebert.encoder.parameters(), 'lr': 2e-5},
    {'params': model.structural_mlp.parameters(), 'lr': 1e-4},
    {'params': model.fusion.parameters(), 'lr': 2e-4}
]
if hasattr(model.codebert, 'pooler'):
    param_groups.append({'params': model.codebert.pooler.parameters(), 'lr': 2e-5})

optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
scaler = torch.amp.GradScaler()

bs = 2
input_ids = torch.randint(0, 1000, (bs, 512)).to(device)
attention_mask = torch.ones(bs, 512).to(device)
struct_feats = torch.randn(bs, 80).to(device)
label = torch.tensor([0, 1]).to(device)

criterion = torch.nn.CrossEntropyLoss()

torch.cuda.reset_peak_memory_stats()
with torch.amp.autocast('cuda'):
    logits = model(input_ids, attention_mask, struct_feats)
    loss = criterion(logits, label)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

alloc = torch.cuda.max_memory_allocated() / (1024**2)
res = torch.cuda.max_memory_reserved() / (1024**2)

print(f"Peak VRAM Alloc: {alloc:.1f} MB")
print(f"Peak VRAM Res: {res:.1f} MB")
