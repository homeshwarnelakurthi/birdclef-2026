from __future__ import annotations
from pathlib import Path
import numpy as np
import onnxruntime as ort

class PerchEmbedder:
    """ONNX wrapper for Google Perch v2 — 1280-dim bird audio embeddings."""
    def __init__(self, onnx_path, num_threads: int = 4):
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"Perch ONNX not found: {self.onnx_path}\n"
                "Download: kaggle datasets download rishikeshjani/perch-onnx-for-birdclef-2026")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.onnx_path), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._threads = num_threads

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        """Run Perch on (batch, 160000) waveforms -> (batch, 1280) embeddings."""
        if waveforms.ndim == 1:
            waveforms = waveforms[np.newaxis, :]
        if waveforms.dtype != np.float32:
            waveforms = waveforms.astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: waveforms})[0]

    @property
    def embedding_dim(self): return 1280
    def __repr__(self): return f"PerchEmbedder(threads={self._threads}, dim=1280)"
