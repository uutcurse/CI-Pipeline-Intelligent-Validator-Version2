import pytest
import torch
import sys
import os
sys.path.append('.')
from src.models.neural_hybrid import NeuralHybrid

def test_hybrid_shapes_and_freezing():
    model = NeuralHybrid(codebert_name='microsoft/codebert-base', structural_model_path=None, freeze_codebert=True)
    
    # 5. frozen CodeBERT parameters
    for name, param in model.codebert.named_parameters():
        assert not param.requires_grad
        
    for name, param in model.structural_mlp.named_parameters():
        assert param.requires_grad
        
    # Mock data
    bs = 2
    input_ids = torch.randint(0, 1000, (bs, 128))
    attention_mask = torch.ones(bs, 128)
    struct_feats = torch.randn(bs, 80)
    
    # 1. CodeBERT embedding shape
    outputs = model.codebert(input_ids, attention_mask)
    text_repr = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
    assert text_repr.shape == (bs, 768)
    
    # 2. structural embedding shape
    _, struct_repr = model.structural_mlp(struct_feats, return_embedding=True)
    assert struct_repr.shape == (bs, 32)
    
    # 3. fusion input shape
    combined = torch.cat([text_repr, struct_repr], dim=1)
    assert combined.shape == (bs, 800)
    assert model.fusion[0].in_features == 800
    
    # 4. logits shape
    logits = model(input_ids=input_ids, attention_mask=attention_mask, structural_features=struct_feats)
    assert logits.shape == (bs, 3)

def test_checkpoint_save_load_consistency(tmp_path):
    model = NeuralHybrid(codebert_name='microsoft/codebert-base')
    model.eval()
    
    bs = 2
    input_ids = torch.randint(0, 100, (bs, 16))
    attention_mask = torch.ones(bs, 16)
    struct_feats = torch.randn(bs, 80)
    
    with torch.no_grad():
        out1 = model(input_ids, attention_mask, struct_feats)
        
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    
    model2 = NeuralHybrid(codebert_name='microsoft/codebert-base')
    model2.load_state_dict(torch.load(path))
    model2.eval()
    
    with torch.no_grad():
        out2 = model2(input_ids, attention_mask, struct_feats)
        
    # 8. checkpoint save/load, 9. inference consistency
    assert torch.allclose(out1, out2, atol=1e-5)
