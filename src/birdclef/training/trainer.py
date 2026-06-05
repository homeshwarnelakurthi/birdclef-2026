"""
EfficientNet-B0 trainer for BirdCLEF+ 2026.

Critical lessons from Kaggle training failures:
1. num_workers MUST be 0 — any value > 0 causes 30GB RAM crash on Kaggle
   Root cause: PyTorch DataLoader workers are child processes that copy full
   parent RAM. With 5.5GB base + 2 workers = 16.5GB+ → OOM.
2. Spectrograms MUST be precomputed to disk as .npy files before training.
   On-the-fly librosa computation is too slow to feed the GPU (27s/batch).
3. Validation batch_size should be same as training to avoid RAM spike.
4. Delete train_loader before creating val_loader to release worker memory.
"""

import gc
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ..models.efficientnet import BirdModel
from .losses import asymmetric_loss


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Mixup augmentation. Improves calibration for rare species."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition metric — macro AUC skipping empty classes."""
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def train_fold(fold: int, df, dataset_class, label_to_idx: dict, n_classes: int = 234):
    """
    Train one fold with DataParallel (2x T4 GPU).

    IMPORTANT: num_workers=0 is mandatory on Kaggle to prevent RAM crash.
    Precomputed spectrograms in df["spec_path"] are required.
    """
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    model = BirdModel(num_classes=n_classes, pretrained=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    best_path = Path("/kaggle/working") / f"best_fold{fold}.pth"

    for epoch in range(1, 4):
        t0 = time.time()

        # Training phase — create and destroy loader each epoch
        train_ds = dataset_class(train_df, mode="train")
        train_loader = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            num_workers=0,       # CRITICAL: must be 0 on Kaggle
            pin_memory=False,
            drop_last=True,
        )

        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Ep{epoch} Train", leave=False):
            x, y = x.cuda(), y.cuda()
            if random.random() < 0.5:
                x, y = mixup_batch(x, y, alpha=0.4)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = asymmetric_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            del x, y

        train_loss /= len(train_loader)
        scheduler.step()

        # CRITICAL: delete train loader before validation to free RAM
        del train_loader, train_ds
        gc.collect()
        torch.cuda.empty_cache()

        # Validation phase
        val_ds = dataset_class(val_df, mode="val")
        val_loader = DataLoader(
            val_ds,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        model.eval()
        n_val = len(val_df)
        val_preds = np.zeros((n_val, n_classes), dtype=np.float32)
        val_labels = np.zeros((n_val, n_classes), dtype=np.float32)
        ptr = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Ep{epoch} Val", leave=False):
                bs = x.size(0)
                with torch.cuda.amp.autocast():
                    preds = torch.sigmoid(model(x.cuda())).cpu().numpy()
                val_preds[ptr: ptr + bs] = preds
                val_labels[ptr: ptr + bs] = y.numpy()
                ptr += bs
                del x, y, preds

        labels_hard = (val_labels > 0.5).astype(np.float32)
        val_auc = macro_auc(labels_hard, val_preds)

        del val_loader, val_ds, val_preds, val_labels, labels_hard
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"Ep {epoch:02d} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | {elapsed:.0f}s")

        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, best_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return val_auc
