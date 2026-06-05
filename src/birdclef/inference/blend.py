"""
Rank-blending and post-processing pipeline.
Final configuration: 60% ProtoSSM / 40% SED + exp002b sidecar.

Key insight: rank averaging (percentile ranks) beats raw probability averaging.
Prevents any single overconfident model from dominating the ensemble.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


EPS = 1e-5


def rank_blend(arrays: list, weights: list) -> np.ndarray:
    """
    Weighted rank averaging across prediction arrays.
    Each array is first converted to percentile ranks before blending.
    """
    assert len(arrays) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6

    ranked = [pd.DataFrame(a).rank(axis=0, pct=True).to_numpy(np.float32) for a in arrays]
    return sum(w * r for w, r in zip(weights, ranked))


def noise_suppression_gate(pred, p_proto, p_sed, weight=0.08):
    """
    Gate 1: If ProtoSSM is confident but SED strongly disagrees, trust ProtoSSM more.
    Reduces false positives from ProtoSSM hallucinations.
    """
    rank_proto = pd.DataFrame(p_proto).rank(axis=0, pct=True).to_numpy(np.float32)
    fake_only = (p_proto > 0.50) & (p_sed < 0.05)
    return np.where(fake_only, (1.0 - weight) * pred + weight * rank_proto, pred)


def temporal_continuity_gate(pred, p_proto, p_sed, file_ids, weight=0.15):
    """
    Gate 2: Fat-tailed t-distribution kernel over 35-second context window.
    Protects continuous calls from being chopped at 5-second boundaries.
    """
    rank_proto = pd.DataFrame(p_proto).rank(axis=0, pct=True).to_numpy(np.float32)

    offs = np.arange(-3, 4, dtype=np.float32)
    kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
    kernel = (kernel / kernel.sum()).astype(np.float32)

    pa_ctx = p_proto.copy()
    for fid in pd.unique(file_ids):
        m = file_ids == fid
        x = p_proto[m]
        if len(x) > 1:
            xp = np.pad(x, ((3, 3), (0, 0)), mode="edge")
            pa_ctx[m] = sum(kernel[i] * xp[i: i + len(x)] for i in range(7))

    xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
    proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12)
    return np.where(
        proto_cont,
        (1.0 - weight) * pred + weight * np.maximum(rank_proto, xctx),
        pred,
    )


def sonotype_mirroring(sub_df, cols):
    """
    Gate 4: Max-pool predictions across acoustically identical insect sonotype groups.

    Biologically motivated: some insect species are acoustically indistinguishable
    and share the same spectrotemporal patterns. If one is detected, all should be.
    """
    MIRROR_PAIRS = (
        ("47158son15", "47158son16"),
        ("47158son09", "47158son12"),
        ("47158son02", "47158son14"),
        ("47158son13", "47158son21", "47158son22", "47158son23"),
    )
    col_to_idx = {l: i for i, l in enumerate(cols)}
    mirror_count = 0

    for group in MIRROR_PAIRS:
        valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
        if len(valid_idx) >= 2:
            group_max = sub_df[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
            for idx in valid_idx:
                sub_df.iloc[:, idx + 1] = group_max
            mirror_count += len(valid_idx)

    return sub_df, mirror_count


def adaptive_rare_class_threshold(sub_df, cols, taxonomy_df):
    """
    Gate 5: Suppress false positives for rare non-bird taxa.

    Amphibia, Mammalia, and Reptilia are systematically over-predicted
    when training data is dominated by Aves (162/234 species).
    """
    rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
    tax_index = taxonomy_df.set_index("primary_label")
    rare_count = 0

    for ci, species in enumerate(cols):
        if species in tax_index.index and tax_index.loc[species, "class_name"] in rare_classes:
            col_idx = ci + 1
            vals = sub_df.iloc[:, col_idx].to_numpy(np.float32)
            thr = vals.mean() + 0.05
            sub_df.iloc[:, col_idx] = np.where(vals < thr, vals * 0.9, vals)
            rare_count += 1

    return sub_df, rare_count
