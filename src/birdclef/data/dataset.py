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
