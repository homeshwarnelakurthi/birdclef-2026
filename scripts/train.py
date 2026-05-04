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
    print(f"\nOOF AUC: {results['oof_auc']:.4f}")
    for i, auc in enumerate(results["fold_aucs"]):
        print(f"  Fold {i+1}: {auc:.4f}")

if __name__ == "__main__":
    main()
