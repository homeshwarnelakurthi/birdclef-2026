"""
Loss functions used in BirdCLEF+ 2026 training.

Key findings:
- Asymmetric Loss (ASL) outperforms standard BCE for weakly-labeled audio
- ASL with gamma_neg=4 heavily down-weights easy negatives (missed annotations)
- Focal loss with gamma=2.0 used in ProtoSSM training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 4,
    gamma_pos: float = 0,
    clip: float = 0.05,
) -> torch.Tensor:
    """
    Asymmetric Loss (ASL) for multi-label classification.

    Addresses the Positive-Unlabeled (PU) problem in BirdCLEF:
    - If a species is not labeled, it doesn't mean it is absent
    - ASL punishes false negatives harder than false positives
    - gamma_neg=4 aggressively down-weights easy negatives
    - gamma_pos=0 treats positives with standard BCE

    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification"
    """
    probs = torch.sigmoid(logits)
    probs_neg = (1.0 - probs).clamp(min=clip)
    probs_pos = probs

    loss_pos = targets * torch.log(probs_pos.clamp(1e-8)) * (1.0 - probs_pos) ** gamma_pos
    loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(1e-8)) * probs_neg ** gamma_neg

    return -(loss_pos + loss_neg).mean()


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss with positive class weighting.
    Used in ProtoSSM training to handle class imbalance.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * bce).mean()
