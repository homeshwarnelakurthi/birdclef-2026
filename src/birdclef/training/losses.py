import torch, torch.nn as nn, torch.nn.functional as F

class BCEWithLabelSmoothing(nn.Module):
    """BCE loss with label smoothing to prevent overconfident predictions."""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, targets):
        t = targets * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(logits, t)

class FocalLoss(nn.Module):
    """Focal loss for rare species (hard example mining)."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets == 1, p, 1 - p)
        aw = torch.where(targets == 1,
                         torch.full_like(targets, self.alpha),
                         torch.full_like(targets, 1 - self.alpha))
        return (aw * (1-pt)**self.gamma * bce).mean()
