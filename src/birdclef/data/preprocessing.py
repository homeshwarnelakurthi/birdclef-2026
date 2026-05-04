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
