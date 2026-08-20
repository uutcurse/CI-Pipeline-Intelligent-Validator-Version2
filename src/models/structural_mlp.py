import torch
import torch.nn as nn

class StructuralMLP(nn.Module):
    def __init__(self, input_dim=80, dropout_rate=0.20):
        super(StructuralMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 32)
        self.act3 = nn.GELU()
        
        self.out = nn.Linear(32, 3)
        
    def forward(self, x, return_embedding=False):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        structural_embedding = self.act3(x)
        
        logits = self.out(structural_embedding)
        
        if return_embedding:
            return logits, structural_embedding
        return logits
