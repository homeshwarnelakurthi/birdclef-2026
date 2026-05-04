# BirdCLEF+ 2026 — Acoustic Species Identification

End-to-end deep learning pipeline for identifying 234 wildlife species from passive
acoustic monitoring recordings in Brazil's Pantanal wetlands.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-lightgrey.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-BirdCLEF%2B%202026-20BEFF.svg)](https://www.kaggle.com/competitions/birdclef-2026)

---

## Overview

Built for the [Kaggle BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
competition. The task is to classify 234 species from five-second audio windows recorded
by passive acoustic monitoring (PAM) devices across the Pantanal wetlands.

| Property | Value |
|---|---|
| Evaluation metric | Macro-averaged ROC-AUC (skipping classes with no positives) |
| Inference constraint | CPU-only, 90-minute runtime limit |
| Audio format | 32 kHz OGG, 5-second windows |
| Training data | ~46,000 recordings from Xeno-canto and iNaturalist |
| Species | 234 (birds, amphibians, insects, mammals, reptiles) |

---

## Results

| Experiment | OOF AUC | LB Score | Key change |
|---|---|---|---|
| Perch v2 + linear probe | 0.891 | 0.909 | Baseline |
| Perch v2 + BirdMLP (4-layer) | 0.921 | 0.921 | Residual blocks, LayerNorm |
| + Label smoothing + focal loss | 0.929 | 0.926 | Rare species handling |
| + Pseudo-label distillation | 0.937 | 0.929 | Soundscape domain adaptation |
| + Prior blending | 0.938 | 0.929 | Species frequency prior |

---

## Architecture

```
Raw audio (.ogg, 32 kHz)
         |
         v
+---------------------+
|    Audio pipeline   |
|  - librosa loading  |
|  - 5s windowing     |
|  - pad / crop       |
|  - augmentation     |
+---------------------+
         |
         v
+---------------------+
|    Perch v2 ONNX    |   Google Bird Vocalization Classifier
|  - 1280-dim output  |   Pre-trained on 80k+ hours of bird audio
|  - CPU optimised    |
+---------------------+
         |
         v
+---------------------+
|      BirdMLP        |   1280 -> 1024 -> 512 -> 256 -> 128 -> 234
|  - LayerNorm + GELU |   Residual connections, Dropout 0.3
|  - Kaiming init     |
+---------------------+
         |
         v
+---------------------+
|  Post-processing    |
|  - Temperature      |   Logit scaling before sigmoid
|  - Prior blending   |   Species frequency prior
+---------------------+
         |
         v
  submission.csv
```

---

## Project Structure

```
birdclef-2026/
+-- configs/
|   +-- base_config.yaml
+-- src/birdclef/
|   +-- data/
|   |   +-- audio.py              Loading, windowing, augmentation
|   |   +-- preprocessing.py      Mel spectrogram, PCEN
|   |   +-- dataset.py            PyTorch Dataset and DataLoader
|   +-- models/
|   |   +-- perch.py              Perch v2 ONNX wrapper
|   |   +-- bird_mlp.py           BirdMLP classifier
|   +-- training/
|   |   +-- trainer.py            5-fold CV training loop
|   |   +-- losses.py             BCE label smoothing, focal loss
|   |   +-- scheduler.py          Cosine annealing with warmup
|   +-- inference/
|   |   +-- pipeline.py           End-to-end inference pipeline
|   +-- utils/
|       +-- config.py             YAML config loader with dataclasses
|       +-- metrics.py            Macro ROC-AUC, per-class AUC
|       +-- logging.py            Structured experiment logger
+-- scripts/
|   +-- train.py
|   +-- predict.py
+-- notebooks/
|   +-- 01_eda.ipynb              Dataset exploration and audio analysis
|   +-- 02_model_analysis.ipynb   OOF analysis, calibration, error analysis
+-- experiments/
+-- requirements.txt
+-- setup.py
```

---

## Skills Demonstrated

| Area | Details |
|---|---|
| Audio ML | librosa, mel spectrograms, PCEN, audio augmentation, PAM data |
| Transfer learning | Google Perch v2 embeddings, frozen backbone + trainable head |
| Deep learning | Custom MLP with residual connections, LayerNorm, mixed precision |
| ONNX | Export, graph optimisation, CPU inference with ONNXRuntime |
| Training pipeline | 5-fold stratified CV, OOF validation, pseudo-label distillation |
| Data engineering | PyTorch Dataset, efficient DataLoader, memory management |
| Experiment tracking | Structured JSON logging, metric history, model versioning |
| Config management | YAML configs with typed dataclasses, reproducible experiments |
| Post-processing | Temperature scaling, prior blending, calibration analysis |

---

## Quickstart

```bash
git clone https://github.com/homeshwarnelakurthi/birdclef-2026.git
cd birdclef-2026
conda create -n birdclef2026 python=3.12 -y
conda activate birdclef2026
pip install -r requirements.txt
pip install -e .
```

```bash
python scripts/train.py --config configs/base_config.yaml --run_name perch_mlp_v1
python scripts/predict.py --model_dir experiments/perch_mlp_v1_20260504 --output submission.csv
```

---

## Key Design Decisions

**Why Perch v2?**
Pre-trained on 80,000+ hours of bird audio. Its 1280-dim embeddings generalise
across species, geographies, and recording conditions — ideal for a 234-class
problem with limited labeled data.

**Why an MLP head?**
The 90-minute CPU-only constraint rules out heavier architectures. A lightweight
MLP on pre-computed embeddings achieves competitive accuracy within the time limit.

**Pseudo-label distillation**
40+ hours of unlabeled Pantanal soundscapes bridge the domain gap between
curated XC/iNat training data and in-the-wild PAM test recordings.

**Temperature scaling**
Dividing logits by T > 1 before sigmoid softens overconfident predictions,
improves calibration, and helps macro-AUC empirically.

---

## Citation

```bibtex
@misc{birdclef2026,
  title  = {BirdCLEF+ 2026: Acoustic Species Identification in the Pantanal},
  author = {Kahl, Stefan and Denton, Tom and Sugai, Larissa and others},
  year   = {2026},
  url    = {https://kaggle.com/competitions/birdclef-2026}
}
```
