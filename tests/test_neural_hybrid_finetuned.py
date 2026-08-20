import pytest
import torch
import sys
sys.path.append('.')
from src.models.neural_hybrid import NeuralHybrid

def test_selective_freezing_and_gradients():
    model = NeuralHybrid(codebert_name='microsoft/codebert-base', freeze_codebert=False, freeze_codebert_embeddings=True)
    
    # 1. frozen embedding layer
    for name, param in model.codebert.embeddings.named_parameters():
        assert not param.requires_grad
        
    # 2. trainable encoder layer
    for name, param in model.codebert.encoder.named_parameters():
        assert param.requires_grad
        
    # 3. trainable structural branch
    for name, param in model.structural_mlp.named_parameters():
        assert param.requires_grad
        
    # 4. trainable fusion head
    for name, param in model.fusion.named_parameters():
        assert param.requires_grad

    # Optimizer discriminative lr logic
    param_groups = [
        {'params': model.codebert.encoder.parameters(), 'lr': 2e-5},
        {'params': model.structural_mlp.parameters(), 'lr': 1e-4},
        {'params': model.fusion.parameters(), 'lr': 2e-4}
    ]
    # In pooler output there is a dense layer that needs training
    if hasattr(model.codebert, 'pooler'):
        param_groups.append({'params': model.codebert.pooler.parameters(), 'lr': 2e-5})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    assert len(optimizer.param_groups) in [3, 4]
    
    # 8. one-step backward produces CodeBERT gradients
    bs = 2
    input_ids = torch.randint(0, 1000, (bs, 16))
    attention_mask = torch.ones(bs, 16)
    struct_feats = torch.randn(bs, 80)
    
    logits = model(input_ids, attention_mask, struct_feats)
    loss = logits.sum()
    loss.backward()
    
    # Check that a representative encoder parameter has gradient
    assert model.codebert.encoder.layer[0].attention.self.query.weight.grad is not None
    # Check that embedding parameter has no gradient
    assert model.codebert.embeddings.word_embeddings.weight.grad is None

def test_fusion_dim_and_logits():
    model = NeuralHybrid(codebert_name='microsoft/codebert-base', freeze_codebert=True)
    assert model.fusion[0].in_features == 800  # 6. 768 + 32 = 800 fusion dimension
    
    bs = 2
    input_ids = torch.randint(0, 100, (bs, 16))
    attention_mask = torch.ones(bs, 16)
    struct_feats = torch.randn(bs, 80)
    
    logits = model(input_ids, attention_mask, struct_feats)
    assert logits.shape == (bs, 3)  # 7. logits shape
