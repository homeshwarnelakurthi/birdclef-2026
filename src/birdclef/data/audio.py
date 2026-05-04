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
