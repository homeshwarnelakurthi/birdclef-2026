from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import List, Dict

def macro_auc_skipping_empty(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact BirdCLEF+ 2026 evaluation metric."""
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else 0.0

def per_class_auc(y_true, y_pred, species_ids: List[str]) -> pd.DataFrame:
    records = []
    for idx, sp in enumerate(species_ids):
        gt = y_true[:, idx]
        n_pos = int(gt.sum())
        auc = roc_auc_score(gt, y_pred[:, idx]) if n_pos > 0 else float("nan")
        records.append({"species": sp, "n_positives": n_pos, "auc": auc})
    return pd.DataFrame(records).dropna().sort_values("auc")

def compute_oof_score(oof_preds, oof_labels) -> Dict:
    n = oof_labels.shape[1]
    aucs, skipped = [], 0
    for i in range(n):
        if oof_labels[:, i].sum() == 0:
            skipped += 1; continue
        try:
            aucs.append(roc_auc_score(oof_labels[:, i], oof_preds[:, i]))
        except ValueError:
            skipped += 1
    return {"macro_auc": float(np.mean(aucs)) if aucs else 0.0,
            "n_classes_scored": len(aucs), "n_classes_skipped": skipped}
