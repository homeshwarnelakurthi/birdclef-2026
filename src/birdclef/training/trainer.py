from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from tqdm import tqdm
from birdclef.data.dataset import TrainDataset, build_dataloader
from birdclef.models.bird_mlp import BirdMLP
from birdclef.models.perch import PerchEmbedder
from birdclef.training.losses import BCEWithLabelSmoothing
from birdclef.training.scheduler import CosineWarmupScheduler
from birdclef.utils.config import Config
from birdclef.utils.logging import ExperimentLogger
from birdclef.utils.metrics import compute_oof_score, macro_auc_skipping_empty

class Trainer:
    """Full 5-fold cross-validation training pipeline."""
    def __init__(self, cfg: Config, perch: PerchEmbedder, logger=None):
        self.cfg = cfg
        self.perch = perch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or ExperimentLogger(cfg.paths.output_dir)
        self.logger.log.info(f"Device: {self.device}")

    def run(self) -> Dict:
        df = pd.read_csv(self.cfg.paths.train_csv)
        tax = pd.read_csv(self.cfg.paths.taxonomy_csv)
        species_to_idx = {sp: i for i, sp in enumerate(tax["primary_label"].tolist())}
        n = len(df)
        oof_preds = np.zeros((n, self.cfg.project.num_classes), np.float32)
        oof_labels = np.zeros((n, self.cfg.project.num_classes), np.float32)
        skf = StratifiedKFold(n_splits=self.cfg.training.num_folds, shuffle=True,
                              random_state=self.cfg.project.seed)
        fold_aucs = []
        for fold, (tr, va) in enumerate(skf.split(df, df["primary_label"])):
            self.logger.log.info(f"\n=== FOLD {fold+1}/{self.cfg.training.num_folds} ===")
            preds, labels = self._train_fold(fold, df, tr, va, species_to_idx)
            oof_preds[va] = preds; oof_labels[va] = labels
            auc = macro_auc_skipping_empty(labels, preds)
            fold_aucs.append(auc)
            self.logger.log_fold(fold+1, auc)
        metrics = compute_oof_score(oof_preds, oof_labels)
        self.logger.log_oof(metrics["macro_auc"], metrics["n_classes_scored"])
        np.save(self.logger.run_dir / "oof_preds.npy", oof_preds)
        np.save(self.logger.run_dir / "oof_labels.npy", oof_labels)
        return {"oof_auc": metrics["macro_auc"], "fold_aucs": fold_aucs}

    def _train_fold(self, fold, df, tr_idx, va_idx, species_to_idx):
        cfg = self.cfg
        train_ds = TrainDataset(df.iloc[tr_idx], cfg.paths.train_audio,
                                species_to_idx, cfg.project.num_classes, augment=True)
        val_ds = TrainDataset(df.iloc[va_idx], cfg.paths.train_audio,
                              species_to_idx, cfg.project.num_classes, augment=False)
        tr_loader = build_dataloader(train_ds, cfg.training.batch_size, True, cfg.training.num_workers)
        va_loader = build_dataloader(val_ds, cfg.training.batch_size*2, False, cfg.training.num_workers)
        model = BirdMLP(cfg.project.num_classes, cfg.model.embedding_dim,
                        cfg.model.hidden_dims, cfg.model.dropout).to(self.device)
        opt = AdamW(model.parameters(), cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        crit = BCEWithLabelSmoothing()
        sched = CosineWarmupScheduler(opt, 3, cfg.training.epochs)
        best_auc = 0.0
        for epoch in range(1, cfg.training.epochs + 1):
            loss = self._train_epoch(model, tr_loader, opt, crit)
            auc, preds, labels = self._val_epoch(model, va_loader)
            sched.step()
            self.logger.log_epoch(epoch, train_loss=loss, val_auc=auc)
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), self.logger.best_model_path(fold))
        model.load_state_dict(torch.load(self.logger.best_model_path(fold), map_location=self.device))
        _, final_preds, final_labels = self._val_epoch(model, va_loader)
        return final_preds, final_labels

    def _train_epoch(self, model, loader, opt, crit):
        model.train(); total = 0.0
        for waves, labels in tqdm(loader, desc="Train", leave=False):
            emb = torch.from_numpy(self.perch.embed(waves.numpy())).to(self.device)
            labels = labels.to(self.device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(emb), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def _val_epoch(self, model, loader):
        model.eval(); all_p, all_l = [], []
        for waves, labels in tqdm(loader, desc="Val", leave=False):
            emb = torch.from_numpy(self.perch.embed(waves.numpy())).to(self.device)
            all_p.append(torch.sigmoid(model(emb)).cpu().numpy())
            all_l.append(labels.numpy())
        p = np.concatenate(all_p); l = np.concatenate(all_l)
        return macro_auc_skipping_empty(l, p), p, l
