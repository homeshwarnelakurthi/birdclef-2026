"""
EfficientNet-B0 with GeM pooling for BirdCLEF+ 2026 mel spectrogram classification.

Architecture choices:
- GeM pooling (p=3) instead of average pooling — better for audio pattern detection
- LayerNorm + GELU in head — more stable than BatchNorm for small batches
- 64 mel bins (reduced from 128) — saves GPU memory with minimal accuracy loss
- 3-channel input — repeat mel spectrogram to match ImageNet pretrained weights

Training approach:
- Asymmetric Loss (ASL) with gamma_neg=4 — handles weak labels
- Mixup with alpha=0.4 — improves calibration for rare species
- CosineAnnealingLR — stable convergence
- Mixed precision (AMP) — 2x speedup on T4
- DataParallel over 2x T4 GPU — effective batch size 128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BirdModel(nn.Module):
    def __init__(self, num_classes: int = 234, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        feat_dim = self.backbone.num_features  # 1280

        # GeM pooling — generalised mean pooling for audio
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def gem_pool(self, x: torch.Tensor) -> torch.Tensor:
        p = self.gem_p.clamp(min=1.0)
        return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(p), 1).pow(1.0 / p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, n_mels, time)
        features = self.backbone.forward_features(x)
        pooled = self.gem_pool(features).flatten(1)
        return self.head(pooled)
