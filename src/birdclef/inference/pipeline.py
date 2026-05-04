from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from birdclef.data.audio import extract_windows, load_audio
from birdclef.models.bird_mlp import BirdMLP
from birdclef.models.perch import PerchEmbedder
from birdclef.utils.config import Config

class InferencePipeline:
    """End-to-end CPU inference: soundscapes -> submission.csv"""
    def __init__(self, cfg: Config, model_dir, species_list, num_folds=5):
        self.cfg = cfg
        self.model_dir = Path(model_dir)
        self.species_list = species_list
        self.num_classes = len(species_list)
        self.perch = PerchEmbedder(cfg.paths.perch_onnx, cfg.inference.onnx_threads)
        self.models = self._load_models(num_folds)

    def run(self, soundscape_dir, output_path="submission.csv", prior=None):
        files = sorted(Path(soundscape_dir).glob("*.ogg"))
        print(f"Processing {len(files)} soundscapes...")
        rows = []
        t0 = time.time()
        for f in tqdm(files):
            rows.extend(self._process(f))
        print(f"Done in {(time.time()-t0)/60:.1f} min")
        df = pd.DataFrame(rows)
        if prior is not None and self.cfg.inference.prior_blend > 0:
            a = self.cfg.inference.prior_blend
            p = np.clip(prior, 1e-6, 1.0); p /= p.max()
            df[self.species_list] = (1-a)*df[self.species_list].values + a*p[np.newaxis,:]
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        return df

    def _process(self, path):
        wav = load_audio(path, self.cfg.audio.sample_rate)
        windows = extract_windows(wav, self.cfg.audio.sample_rate,
                                  self.cfg.project.window_duration,
                                  self.cfg.project.window_duration)
        if not windows: return []
        emb = torch.from_numpy(self.perch.embed(np.stack(windows)))
        probs = self._predict(emb)
        stem = path.stem
        return [{"row_id": f"{stem}_{int((i+1)*self.cfg.project.window_duration)}",
                 **dict(zip(self.species_list, p.tolist()))}
                for i, p in enumerate(probs)]

    @torch.no_grad()
    def _predict(self, emb):
        all_p = []
        for m in self.models:
            all_p.append(torch.sigmoid(m(emb) / self.cfg.inference.temperature).numpy())
        return np.mean(all_p, axis=0)

    def _load_models(self, num_folds):
        models = []
        for fold in range(num_folds):
            p = self.model_dir / f"checkpoints/best_fold{fold}.pt"
            if not p.exists(): continue
            m = BirdMLP(self.num_classes, self.cfg.model.embedding_dim,
                        self.cfg.model.hidden_dims, dropout=0.0)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval(); models.append(m)
        print(f"Loaded {len(models)} fold models")
        return models
