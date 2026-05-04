from __future__ import annotations
from typing import List, Optional
import torch, torch.nn as nn

class MLPBlock(nn.Module):
    """Linear -> LayerNorm -> GELU -> Dropout with optional residual."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = in_dim == out_dim

    def forward(self, x):
        out = self.drop(self.act(self.norm(self.linear(x))))
        return out + x if self.residual else out

class BirdMLP(nn.Module):
    """Multi-layer perceptron: Perch embeddings -> 234 species logits."""
    def __init__(self, num_classes=234, embedding_dim=1280,
                 hidden_dims: Optional[List[int]] = None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]
        dims = [embedding_dim] + hidden_dims
        self.body = nn.Sequential(*[
            MLPBlock(dims[i], dims[i+1], dropout) for i in range(len(dims)-1)
        ])
        self.head = nn.Linear(hidden_dims[-1], num_classes)
        self._init_weights()

    def forward(self, x):
        return self.head(self.body(x))

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return f"BirdMLP(params={self.count_parameters():,})"
