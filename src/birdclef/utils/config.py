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
