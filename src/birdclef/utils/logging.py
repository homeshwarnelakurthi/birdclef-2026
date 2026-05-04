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
            f.write(json.dumps(record) + "\n")
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
