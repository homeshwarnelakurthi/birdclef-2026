"""
Precompute mel spectrograms to disk as .npy files.

This is MANDATORY for stable training on Kaggle T4 GPU.

Why this matters:
- On-the-fly librosa computation: ~9s per batch, CPU at 400%, GPU idle 94%
- Precomputed .npy loading: ~0.5s per batch, stable RAM, GPU fully utilized
- With num_workers > 0: 30GB RAM crash guaranteed (workers copy parent RAM)
- With num_workers=0 + .npy files: stable ~8GB RAM throughout training

Usage:
    python scripts/precompute_spectrograms.py \
        --data_dir /kaggle/input/competitions/birdclef-2026/train_audio \
        --output_dir /kaggle/working/specs \
        --train_csv /kaggle/input/competitions/birdclef-2026/train.csv
"""

import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm


def compute_mel(filepath: str, sr: int = 32000, target_samples: int = 160000) -> np.ndarray:
    try:
        y, orig_sr = sf.read(filepath, dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    except Exception:
        y = np.zeros(target_samples, dtype=np.float32)

    if len(y) < target_samples:
        y = np.tile(y, (target_samples // len(y)) + 1)
    y = y[:target_samples].astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=320,
        n_mels=64, fmin=50, fmax=14000, power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    mel = (mel + 80.0) / 80.0
    return mel.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_csv", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    df = df[(df["rating"] >= 4) | (df["rating"] == 0)].reset_index(drop=True)

    existing = sum(1 for i in range(len(df)) if (output_dir / f"{i}.npy").exists())
    print(f"Files to compute: {len(df) - existing} (already done: {existing})")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        out_path = output_dir / f"{i}.npy"
        if out_path.exists():
            continue
        filepath = str(Path(args.data_dir) / row["filename"])
        mel = compute_mel(filepath)
        np.save(out_path, mel)

    print(f"Done. Total: {len(list(output_dir.glob('*.npy')))} files")


if __name__ == "__main__":
    main()
