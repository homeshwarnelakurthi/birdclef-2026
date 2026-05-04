import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    """Linear warmup + cosine annealing LR schedule."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / max(1, self.warmup_epochs)
        else:
            p = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
        return [lr * scale for lr in self.base_lrs]
