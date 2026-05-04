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
