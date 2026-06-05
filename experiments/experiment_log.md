# Experiment Log — BirdCLEF+ 2026

Final result: 341st / 4,243 — Bronze Medal (Top 8%)

## EXP-001 — Perch + ProtoSSM baseline
- Score: 0.909 | ProtoSSM d_model=128, 2 SSM layers, 60/40 blend

## EXP-002 — TTA addition
- Score: 0.943 | TTA shifts [0,1,-1,2,-2], rank averaging | +0.034 gain

## EXP-003 — Sonotype mirroring + rare-class thresholding
- Score: 0.947 | 10 sonotype columns max-pooled, 44 rare species suppressed

## EXP-004 — Temporal continuity gate
- Score: 0.947 | Fat-tailed t-kernel, 35s context window

## EXP-005 — BirdNET 3-way blend
- Score: 0.946 (regression) | BirdNET diluted 170/234 species. Removed.

## EXP-006 — wslll CSV blend
- Score: Scoring error | Row IDs mismatch train vs test soundscapes

## EXP-007 — Beta49 (pre-trained ProtoSSM weights)
- Score: 0.949 | sgkfk-202604041716 weights, d_model=256, 3 layers

## EXP-008 — Pseudo-labeling on 200 soundscapes
- Score: 0.900 (major regression) | Density 0.6465 — biologically impossible

## EXP-009 — EoS8 + exp002b PCEN/ConvNeXt sidecar
- Score: 0.950 | OOF-Gated blend + taxonomy smoothing | FINAL SUBMISSION

## EXP-010 to EXP-013 — EfficientNet training attempts
- All failed due to Kaggle RAM crashes (30GB limit with DataLoader workers)
- Root cause: num_workers > 0 creates child processes copying full parent RAM
- Fix confirmed: precomputed .npy spectrograms + num_workers=0

## Final private LB: 341st / 4,243 — Bronze Medal
