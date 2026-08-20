import torch
import torch.nn as nn
from transformers import AutoModel
import sys
sys.path.append('.')
from src.models.structural_mlp import StructuralMLP

class NeuralHybrid(nn.Module):
    def __init__(self, codebert_name='microsoft/codebert-base', structural_model_path=None, freeze_codebert=True, freeze_codebert_embeddings=False, use_text=True, use_structure=True):
        super(NeuralHybrid, self).__init__()
        self.use_text = use_text
        self.use_structure = use_structure
        
        if self.use_text:
            self.codebert = AutoModel.from_pretrained(codebert_name, local_files_only=True)
            if freeze_codebert:
                for param in self.codebert.parameters():
                    param.requires_grad = False
            elif freeze_codebert_embeddings:
                for param in self.codebert.embeddings.parameters():
                    param.requires_grad = False
                    
        if self.use_structure:
            self.structural_mlp = StructuralMLP(input_dim=80)
            if structural_model_path is not None:
                self.structural_mlp.load_state_dict(torch.load(structural_model_path, map_location='cpu'))
                
        fusion_dim = 0
        if self.use_text: fusion_dim += 768
        if self.use_structure: fusion_dim += 32
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 3)
        )
        
    def forward(self, input_ids=None, attention_mask=None, structural_features=None):
        features = []
        if self.use_text:
            outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                text_repr = outputs.pooler_output
            else:
                text_repr = outputs.last_hidden_state[:, 0, :]
            features.append(text_repr)
            
        if self.use_structure:
            _, struct_repr = self.structural_mlp(structural_features, return_embedding=True)
            features.append(struct_repr)
            
        combined = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        logits = self.fusion(combined)
        return logits
