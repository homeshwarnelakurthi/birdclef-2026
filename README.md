# BirdCLEF+ 2026 — Acoustic Species Identification in the Pantanal

**🏅 Final Result: 341st place out of 4,243 teams — Bronze Medal (Top 8%)**

Kaggle competition: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

---

## Competition Overview

Identify 234 wildlife species (birds, amphibians, mammals, reptiles, insects) from passive acoustic monitoring recordings collected across Brazil's Pantanal wetlands. Evaluation metric: macro-averaged ROC-AUC. Constraint: CPU-only inference, 90-minute runtime limit.

---

## Final Results

| Metric | Value |
|---|---|
| Public LB score | 0.950 |
| Private LB rank | **341st / 4,243** |
| Medal | **Bronze (Top 8%)** |

The jump from ~935th public to 341st private demonstrates strong generalization — the pipeline was not over-fitted to the public test set.

---

## Pipeline Architecture

```
Test soundscapes (.ogg, 60s, 32kHz)
          │
          ▼
┌─────────────────────────┐
│   Perch v2 ONNX         │  1536-dim embeddings per 5s window
└───────────┬─────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐   ┌──────────────────┐
│ProtoSSM │   │  SED 5-fold ONNX │
│d_model= │   │  (Tucker Arrants)│
│256, 3   │   │  256 mel bins    │
│SSM      │   │  clip+frame      │
│layers,  │   │  logits, 5 folds │
│TTA x5   │   └────────┬─────────┘
└────┬────┘            │
     └────────┬────────┘
              ▼
    ┌─────────────────────────┐
    │  Rank blend 60/40       │
    └───────────┬─────────────┘
                ▼
    ┌─────────────────────────────────┐
    │  exp002b ConvNeXt sidecar       │
    │  OOF-Gated: B = A + W·M·(S-A)  │
    └───────────┬─────────────────────┘
                ▼
    ┌─────────────────────────┐
    │  Post-processing        │
    │  Taxonomy smoothing     │
    │  Sonotype mirroring     │
    │  Rare-class threshold   │
    │  Temporal gates         │
    └───────────┬─────────────┘
                ▼
         submission.csv
```

---

## Leaderboard Progression

| Version | Score | Key change |
|---|---|---|
| v1 Perch baseline | 0.909 | ProtoSSM + SED 2-way blend |
| v2 Better blend | 0.943 | Rank averaging, TTA 5 shifts |
| v3 Post-processing | 0.947 | Sonotype mirroring, rare-class thresholding, temporal gates |
| v4 BirdNET blend | 0.946 | BirdNET diluted signal — regressed |
| v5 Clean 2-way | 0.947 | Reverted to optimal 60/40 |
| v6 EoS8 sidecar | **0.950** | exp002b ConvNeXt sidecar + taxonomy smoothing |
| **Private LB** | **Bronze** | **341st / 4,243** |

---

## Models Used

### Backbone
| Model | Source | Role |
|---|---|---|
| Perch v2 ONNX | Google DeepMind | Audio embeddings 1536-dim |
| Perch v2 TF SavedModel | Google DeepMind | Fallback |

### Sequence Models
| Model | Parameters | Role |
|---|---|---|
| LightProtoSSM v4 | 5,776,250 | Temporal classification over 12 windows |
| ResidualSSM | ~200k | Residual correction of first-pass predictions |
| MLP probes (58 species) | Small per-species MLPs | PCA-64 embedding fine-tuning |

### External Models
| Model | Source | Standalone score |
|---|---|---|
| SED 5-fold ONNX | Tucker Arrants | ~0.929 |
| BirdNET v2.4 TFLite | Cornell Lab | ~0.862 |
| exp002b ConvNeXt | Pilkwang Kim | Sidecar correction only |

---

## Techniques Tried

### What worked
- Rank-averaging ensemble (percentile ranks, not raw probabilities)
- TTA — 5 circular temporal shifts `[0, 1, -1, 2, -2]`
- Sonotype mirroring — max-pool across insect sonotype groups
- Adaptive rare-class thresholding (Amphibia, Mammalia, Reptilia)
- Temporal continuity gate — fat-tailed kernel over 35s context
- Site + hour prior tables — joint site-hour bucket priors
- Isotonic calibration — per-class threshold optimization on OOF
- SWA (Stochastic Weight Averaging) in ProtoSSM training
- Mixup + focal loss in ProtoSSM training
- Taxonomy smoothing — genus and class level soft propagation
- OOF-Gated sidecar blending — per-class gate from OOF predictions
- Asymmetric loss (ASL) for EfficientNet (gamma_neg=4, gamma_pos=0)
- Precomputed mel spectrograms to avoid RAM crashes

### What did not work
- BirdNET global blend — only 64/234 covered; diluted predictions
- wslll notebook CSV blend — row_ids from train not test soundscapes
- Pseudo-labeling at density 0.64 — crashed score to 0.900
- PCEN preprocessing — ~1628s/epoch CPU bottleneck
- EfficientNet training with DataLoader workers — 30GB RAM crash
- Increasing num_workers beyond 0 on Kaggle — always OOM

---

## Frameworks and Technologies

| Category | Technology |
|---|---|
| Deep learning | PyTorch 2.8.0+cpu / 2.10.0+cu128 |
| Audio backbone | TensorFlow 2.20.0 (Perch) |
| ONNX inference | ONNXRuntime 1.24.4 |
| Audio processing | librosa 0.10.x, soundfile |
| ML utilities | scikit-learn (PCA, IsotonicRegression, MLPClassifier) |
| Model architectures | timm 1.0.25 (EfficientNet-B0) |
| Data processing | pandas, numpy |
| Version control | Git + GitHub |
| Training platform | Kaggle T4 x2 GPU (30h/week quota) |
| Local development | VS Code, Windows 11, Python 3.12.7 Anaconda |
| Submission platform | Kaggle CPU notebook, 90-min limit |
| AI assistant | Claude (Anthropic) — used throughout competition |

---

## Repository Structure

```
birdclef-2026/
├── configs/
│   └── base_config.yaml
├── data/
│   ├── dataset.py
│   ├── augmentation.py
│   └── preprocessing.py
├── models/
│   ├── proto_ssm.py
│   ├── residual_ssm.py
│   ├── efficientnet.py
│   └── mlp_probes.py
├── inference/
│   ├── perch_onnx.py
│   ├── sed_inference.py
│   └── blend.py
├── training/
│   ├── train_proto_ssm.py
│   └── train_efficientnet.py
├── utils/
│   ├── metrics.py
│   ├── prior_tables.py
│   └── calibration.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_analysis.ipynb
├── scripts/
│   ├── build_cache.py
│   └── precompute_spectrograms.py
├── experiments/
│   └── experiment_log.md
├── requirements.txt
├── setup.py
└── README.md
```

---

## Key Datasets (Kaggle)

| Dataset | Purpose |
|---|---|
| `competitions/birdclef-2026` | Train audio, soundscapes, labels |
| `rishikeshjani/perch-onnx-for-birdclef-2026` | Perch ONNX fast inference |
| `tuckerarrants/bc2026-distilled-sed-public` | Distilled SED 5-fold ONNX |
| `jaejohn/perch-meta` | Pre-computed embeddings for 59 soundscapes |
| `hideyukizushi/sgkfk-202604041716` | Pre-trained ProtoSSM + ResidualSSM |
| `pilkwang/birdclef26-sidecar-exp002b-5s-weakaudio` | ConvNeXt PCEN sidecar |

---

## Lessons Learned

1. Rank averaging beats probability averaging — prevents overconfident models from dominating
2. CPU inference is the real constraint — every design decision must fit 90 minutes
3. Diversity matters more than raw performance — SED + ProtoSSM blend better than two ProtoSSMs
4. Private LB generalization beats public LB optimization — robust post-processing wins
5. `num_workers=0` with precomputed spectrograms is the only stable Kaggle training setup
6. Pseudo-labeling requires surgical precision — density above 0.1 is biologically impossible
7. Taxonomy smoothing is free performance — genus + class propagation with no runtime cost

---

## Citation

```
Stefan Kahl, Tom Denton, Larissa Sugai, Liliana Piatti, Ryan Holbrook,
Holger Klinck, and Ashley Oldacre. BirdCLEF+ 2026.
https://kaggle.com/competitions/birdclef-2026, 2026. Kaggle.
```
