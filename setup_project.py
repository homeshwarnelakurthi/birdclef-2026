import os, textwrap
from pathlib import Path

def w(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(textwrap.dedent(content).lstrip(), encoding='utf-8')
    print(f"  created: {path}")

print("Creating all project files...")

w("setup.py", """
    from setuptools import setup, find_packages
    setup(
        name="birdclef",
        version="1.0.0",
        description="Acoustic species identification — BirdCLEF+ 2026",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        python_requires=">=3.10",
    )
""")

w(".gitignore", """
    __pycache__/
    *.py[cod]
    *.egg-info/
    .env
    data/raw/
    *.ogg *.mp3 *.wav
    *.pt *.pth *.onnx
    .DS_Store
    Thumbs.db
    *.log
""")

w("configs/base_config.yaml", """
    project:
      name: birdclef-2026
      seed: 42
      num_classes: 234
      sample_rate: 32000
      window_duration: 5
    paths:
      data_root: data/raw
      train_audio: data/raw/train_audio
      test_soundscapes: data/raw/test_soundscapes
      train_csv: data/raw/train.csv
      taxonomy_csv: data/raw/taxonomy.csv
      output_dir: experiments
      perch_onnx: models/perch_v2.onnx
    audio:
      sample_rate: 32000
      n_mels: 128
      fmin: 50
      fmax: 14000
      target_length: 160000
    training:
      epochs: 30
      batch_size: 64
      learning_rate: 0.001
      weight_decay: 0.0001
      num_folds: 5
      dropout: 0.3
    model:
      embedding_dim: 1280
      hidden_dims: [1024, 512, 256, 128]
      dropout: 0.3
    inference:
      batch_size: 32
      temperature: 1.5
      prior_blend: 0.2
      onnx_threads: 4
""")

w("src/birdclef/__init__.py", '"""BirdCLEF+ 2026 — acoustic species identification pipeline."""\n')
w("src/birdclef/data/__init__.py", "")
w("src/birdclef/models/__init__.py", "")
w("src/birdclef/training/__init__.py", "")
w("src/birdclef/inference/__init__.py", "")
w("src/birdclef/utils/__init__.py", "")

w("src/birdclef/utils/config.py", '''
    from __future__ import annotations
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import List
    import random, numpy as np, torch, yaml

    @dataclass
    class ProjectConfig:
        name: str = "birdclef-2026"
        seed: int = 42
        num_classes: int = 234
        sample_rate: int = 32000
        window_duration: int = 5

    @dataclass
    class PathConfig:
        data_root: str = "data/raw"
        train_audio: str = "data/raw/train_audio"
        test_soundscapes: str = "data/raw/test_soundscapes"
        train_csv: str = "data/raw/train.csv"
        taxonomy_csv: str = "data/raw/taxonomy.csv"
        output_dir: str = "experiments"
        perch_onnx: str = "models/perch_v2.onnx"

    @dataclass
    class AudioConfig:
        sample_rate: int = 32000
        n_mels: int = 128
        fmin: float = 50.0
        fmax: float = 14000.0
        target_length: int = 160000

    @dataclass
    class TrainingConfig:
        epochs: int = 30
        batch_size: int = 64
        learning_rate: float = 1e-3
        weight_decay: float = 1e-4
        num_folds: int = 5
        dropout: float = 0.3
        num_workers: int = 4

    @dataclass
    class ModelConfig:
        embedding_dim: int = 1280
        hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256, 128])
        dropout: float = 0.3

    @dataclass
    class InferenceConfig:
        batch_size: int = 32
        temperature: float = 1.5
        prior_blend: float = 0.2
        onnx_threads: int = 4

    @dataclass
    class Config:
        project: ProjectConfig = field(default_factory=ProjectConfig)
        paths: PathConfig = field(default_factory=PathConfig)
        audio: AudioConfig = field(default_factory=AudioConfig)
        training: TrainingConfig = field(default_factory=TrainingConfig)
        model: ModelConfig = field(default_factory=ModelConfig)
        inference: InferenceConfig = field(default_factory=InferenceConfig)

    def load_config(config_path: str) -> Config:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        return Config(
            project=ProjectConfig(**raw.get("project", {})),
            paths=PathConfig(**raw.get("paths", {})),
            audio=AudioConfig(**raw.get("audio", {})),
            training=TrainingConfig(**raw.get("training", {})),
            model=ModelConfig(**raw.get("model", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
        )

    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
''')

w("src/birdclef/utils/metrics.py", '''
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
''')

w("src/birdclef/utils/logging.py", '''
    from __future__ import annotations
    import json, logging
    from datetime import datetime
    from pathlib import Path
    from typing import Any, Dict, Optional
    import yaml

    class ExperimentLogger:
        def __init__(self, output_dir: str, run_name: Optional[str] = None) -> None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{run_name}_{ts}" if run_name else ts
            self.run_dir = Path(output_dir) / name
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "checkpoints").mkdir(exist_ok=True)
            self.metrics_file = self.run_dir / "metrics.jsonl"
            self.history = []
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.FileHandler(self.run_dir / "run.log"), logging.StreamHandler()],
            )
            self.log = logging.getLogger(name)
            self.log.info(f"Run dir: {self.run_dir}")

        def log_epoch(self, epoch: int, **metrics: float) -> None:
            record = {"epoch": epoch, **metrics}
            self.history.append(record)
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(record) + "\\n")
            self.log.info("Epoch %03d | %s", epoch,
                          "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        def log_fold(self, fold: int, val_auc: float) -> None:
            self.log.info(f"Fold {fold} | val_auc={val_auc:.4f}")

        def log_oof(self, oof_auc: float, n_scored: int) -> None:
            self.log.info(f"OOF AUC={oof_auc:.4f} | classes_scored={n_scored}")

        def save_config(self, cfg_dict: Dict) -> None:
            with open(self.run_dir / "config.yaml", "w") as f:
                yaml.dump(cfg_dict, f)

        def best_model_path(self, fold: int) -> Path:
            return self.run_dir / "checkpoints" / f"best_fold{fold}.pt"
''')

w("src/birdclef/data/audio.py", '''
    from __future__ import annotations
    import random
    from pathlib import Path
    from typing import List, Tuple
    import librosa, numpy as np

    def load_audio(path, target_sr: int = 32000) -> np.ndarray:
        """Load audio file and resample to target_sr."""
        waveform, _ = librosa.load(str(path), sr=target_sr, mono=True, dtype=np.float32)
        return waveform

    def pad_or_crop(waveform: np.ndarray, target_length: int, mode: str = "random") -> np.ndarray:
        """Pad by tiling or crop to exact target_length."""
        n = len(waveform)
        if n == target_length:
            return waveform
        if n < target_length:
            waveform = np.tile(waveform, (target_length // n) + 1)
        excess = len(waveform) - target_length
        start = random.randint(0, excess) if mode == "random" else excess // 2
        return waveform[start: start + target_length]

    def extract_windows(waveform: np.ndarray, sr: int = 32000,
                        window_s: float = 5.0, hop_s: float = 5.0) -> List[np.ndarray]:
        """Split a soundscape into fixed-length windows."""
        ws = int(window_s * sr)
        hs = int(hop_s * sr)
        windows = []
        start = 0
        while start + ws <= len(waveform):
            windows.append(waveform[start: start + ws])
            start += hs
        if start < len(waveform):
            windows.append(pad_or_crop(waveform[start:], ws, mode="center"))
        return windows

    def add_gaussian_noise(waveform: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        sig_pow = np.mean(waveform ** 2)
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        return waveform + np.random.randn(len(waveform)).astype(np.float32) * np.sqrt(noise_pow)

    def time_shift(waveform: np.ndarray, max_frac: float = 0.1) -> np.ndarray:
        shift = random.randint(0, int(len(waveform) * max_frac))
        return np.roll(waveform, shift)

    def gain_augment(waveform: np.ndarray, db_range: Tuple = (-6.0, 6.0)) -> np.ndarray:
        gain = 10 ** (random.uniform(*db_range) / 20)
        return np.clip(waveform * gain, -1.0, 1.0)

    def mixup(wa, wb, la, lb, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        return (lam*wa + (1-lam)*wb).astype(np.float32), (lam*la + (1-lam)*lb).astype(np.float32)

    def apply_train_augmentations(waveform: np.ndarray) -> np.ndarray:
        if random.random() < 0.3:
            waveform = add_gaussian_noise(waveform, snr_db=random.uniform(15, 30))
        if random.random() < 0.3:
            waveform = time_shift(waveform)
        if random.random() < 0.5:
            waveform = gain_augment(waveform)
        return waveform
''')

w("src/birdclef/data/preprocessing.py", '''
    from __future__ import annotations
    import numpy as np
    import librosa
    from typing import Optional

    def waveform_to_melspec(waveform: np.ndarray, sr: int = 32000,
                            n_fft: int = 1024, hop_length: int = 320,
                            n_mels: int = 128, fmin: float = 50.0,
                            fmax: float = 14000.0) -> np.ndarray:
        """Convert waveform to log-mel spectrogram (dB scale)."""
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
        return librosa.power_to_db(mel, ref=np.max, top_db=80.0).astype(np.float32)

    def normalize_spectrogram(spec: np.ndarray, mean=None, std=None, eps=1e-6) -> np.ndarray:
        m = mean if mean is not None else spec.mean()
        s = std if std is not None else spec.std()
        return (spec - m) / (s + eps)

    def pcen(waveform: np.ndarray, sr: int = 32000,
             hop_length: int = 320, n_mels: int = 128,
             fmin: float = 50.0, fmax: float = 14000.0) -> np.ndarray:
        """Per-Channel Energy Normalization — robust to background noise in PAM data."""
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0)
        return librosa.pcen(mel * (2**31), sr=sr, hop_length=hop_length).astype(np.float32)
''')

w("src/birdclef/data/dataset.py", '''
    from __future__ import annotations
    from pathlib import Path
    from typing import Dict, List, Tuple
    import numpy as np, pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from birdclef.data.audio import apply_train_augmentations, load_audio, pad_or_crop

    class TrainDataset(Dataset):
        """Short labelled XC/iNat recordings with multi-hot labels."""
        def __init__(self, df, audio_dir, species_to_idx, num_classes=234,
                     sample_rate=32000, target_length=160000, augment=False):
            self.df = df.reset_index(drop=True)
            self.audio_dir = Path(audio_dir)
            self.species_to_idx = species_to_idx
            self.num_classes = num_classes
            self.sample_rate = sample_rate
            self.target_length = target_length
            self.augment = augment

        def __len__(self): return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            waveform = load_audio(self.audio_dir / row["filename"], self.sample_rate)
            waveform = pad_or_crop(waveform, self.target_length, "random")
            if self.augment:
                waveform = apply_train_augmentations(waveform)
            label = np.zeros(self.num_classes, dtype=np.float32)
            if row["primary_label"] in self.species_to_idx:
                label[self.species_to_idx[row["primary_label"]]] = 1.0
            sec = row.get("secondary_labels", "")
            if isinstance(sec, str) and sec:
                for sp in sec.split():
                    if sp in self.species_to_idx:
                        label[self.species_to_idx[sp]] = 0.5
            return torch.from_numpy(waveform), torch.from_numpy(label)

    def build_dataloader(dataset, batch_size=64, shuffle=False, num_workers=4):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, drop_last=shuffle)
''')

w("src/birdclef/models/perch.py", '''
    from __future__ import annotations
    from pathlib import Path
    import numpy as np
    import onnxruntime as ort

    class PerchEmbedder:
        """ONNX wrapper for Google Perch v2 — 1280-dim bird audio embeddings."""
        def __init__(self, onnx_path, num_threads: int = 4):
            self.onnx_path = Path(onnx_path)
            if not self.onnx_path.exists():
                raise FileNotFoundError(
                    f"Perch ONNX not found: {self.onnx_path}\\n"
                    "Download: kaggle datasets download rishikeshjani/perch-onnx-for-birdclef-2026")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = num_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.onnx_path), opts,
                                                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self._threads = num_threads

        def embed(self, waveforms: np.ndarray) -> np.ndarray:
            """Run Perch on (batch, 160000) waveforms -> (batch, 1280) embeddings."""
            if waveforms.ndim == 1:
                waveforms = waveforms[np.newaxis, :]
            if waveforms.dtype != np.float32:
                waveforms = waveforms.astype(np.float32)
            return self.session.run([self.output_name], {self.input_name: waveforms})[0]

        @property
        def embedding_dim(self): return 1280
        def __repr__(self): return f"PerchEmbedder(threads={self._threads}, dim=1280)"
''')

w("src/birdclef/models/bird_mlp.py", '''
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
''')

w("src/birdclef/training/losses.py", '''
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
''')

w("src/birdclef/training/scheduler.py", '''
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
''')

w("src/birdclef/training/trainer.py", '''
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
                self.logger.log.info(f"\\n=== FOLD {fold+1}/{self.cfg.training.num_folds} ===")
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
''')

w("src/birdclef/inference/pipeline.py", '''
    from __future__ import annotations
    import time
    from pathlib import Path
    from typing import Dict, List, Optional
    import numpy as np, pandas as pd, torch
    from tqdm import tqdm
    from birdclef.data.audio import extract_windows, load_audio
    from birdclef.models.bird_mlp import BirdMLP
    from birdclef.models.perch import PerchEmbedder
    from birdclef.utils.config import Config

    class InferencePipeline:
        """End-to-end CPU inference: soundscapes -> submission.csv"""
        def __init__(self, cfg: Config, model_dir, species_list, num_folds=5):
            self.cfg = cfg
            self.model_dir = Path(model_dir)
            self.species_list = species_list
            self.num_classes = len(species_list)
            self.perch = PerchEmbedder(cfg.paths.perch_onnx, cfg.inference.onnx_threads)
            self.models = self._load_models(num_folds)

        def run(self, soundscape_dir, output_path="submission.csv", prior=None):
            files = sorted(Path(soundscape_dir).glob("*.ogg"))
            print(f"Processing {len(files)} soundscapes...")
            rows = []
            t0 = time.time()
            for f in tqdm(files):
                rows.extend(self._process(f))
            print(f"Done in {(time.time()-t0)/60:.1f} min")
            df = pd.DataFrame(rows)
            if prior is not None and self.cfg.inference.prior_blend > 0:
                a = self.cfg.inference.prior_blend
                p = np.clip(prior, 1e-6, 1.0); p /= p.max()
                df[self.species_list] = (1-a)*df[self.species_list].values + a*p[np.newaxis,:]
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
            return df

        def _process(self, path):
            wav = load_audio(path, self.cfg.audio.sample_rate)
            windows = extract_windows(wav, self.cfg.audio.sample_rate,
                                      self.cfg.project.window_duration,
                                      self.cfg.project.window_duration)
            if not windows: return []
            emb = torch.from_numpy(self.perch.embed(np.stack(windows)))
            probs = self._predict(emb)
            stem = path.stem
            return [{"row_id": f"{stem}_{int((i+1)*self.cfg.project.window_duration)}",
                     **dict(zip(self.species_list, p.tolist()))}
                    for i, p in enumerate(probs)]

        @torch.no_grad()
        def _predict(self, emb):
            all_p = []
            for m in self.models:
                all_p.append(torch.sigmoid(m(emb) / self.cfg.inference.temperature).numpy())
            return np.mean(all_p, axis=0)

        def _load_models(self, num_folds):
            models = []
            for fold in range(num_folds):
                p = self.model_dir / f"checkpoints/best_fold{fold}.pt"
                if not p.exists(): continue
                m = BirdMLP(self.num_classes, self.cfg.model.embedding_dim,
                            self.cfg.model.hidden_dims, dropout=0.0)
                m.load_state_dict(torch.load(p, map_location="cpu"))
                m.eval(); models.append(m)
            print(f"Loaded {len(models)} fold models")
            return models
''')

w("scripts/train.py", '''
    """Entry point: python scripts/train.py"""
    from __future__ import annotations
    import argparse, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from birdclef.models.perch import PerchEmbedder
    from birdclef.training.trainer import Trainer
    from birdclef.utils.config import load_config, set_seed
    from birdclef.utils.logging import ExperimentLogger

    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--config", default="configs/base_config.yaml")
        p.add_argument("--run_name", default="perch_mlp")
        args = p.parse_args()
        cfg = load_config(args.config)
        set_seed(cfg.project.seed)
        logger = ExperimentLogger(cfg.paths.output_dir, args.run_name)
        logger.save_config(cfg.__dict__)
        perch = PerchEmbedder(cfg.paths.perch_onnx, cfg.inference.onnx_threads)
        trainer = Trainer(cfg, perch, logger)
        results = trainer.run()
        print(f"\\nOOF AUC: {results[\'oof_auc\']:.4f}")
        for i, auc in enumerate(results["fold_aucs"]):
            print(f"  Fold {i+1}: {auc:.4f}")

    if __name__ == "__main__":
        main()
''')

w("scripts/predict.py", '''
    """Entry point: python scripts/predict.py --model_dir experiments/RUN_NAME"""
    from __future__ import annotations
    import argparse, sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from birdclef.inference.pipeline import InferencePipeline
    from birdclef.utils.config import load_config

    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--config", default="configs/base_config.yaml")
        p.add_argument("--model_dir", required=True)
        p.add_argument("--output", default="submission.csv")
        p.add_argument("--num_folds", type=int, default=5)
        args = p.parse_args()
        cfg = load_config(args.config)
        tax = pd.read_csv(cfg.paths.taxonomy_csv)
        species_list = tax["primary_label"].tolist()
        pipeline = InferencePipeline(cfg, args.model_dir, species_list, args.num_folds)
        pipeline.run(cfg.paths.test_soundscapes, args.output)

    if __name__ == "__main__":
        main()
''')

w("experiments/.gitkeep", "")
w("notebooks/.gitkeep", "")

print("\\n✅ All files created successfully!")
print("\\nProject structure:")
for root, dirs, files_ in os.walk("."):
    dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", "data"]]
    level = root.replace(".", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files_:
        print(f"{indent}  {f}")
