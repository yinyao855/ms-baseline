"""SBERT-based class semantic embedding.

Reuses the model recommended by the MONO2REST paper
(``sentence-transformers/bert-base-nli-mean-tokens``) and falls back to a
deterministic hash embedding if the model cannot be loaded — this keeps
the offline test path working without forcing a 420MB download.
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from .data_models import ClassNode


class ClassEmbedder:
    def __init__(self, model_name: str = "bert-base-nli-mean-tokens"):
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print(
                "[nsga3] sentence-transformers not installed — "
                "falling back to hash-based embedding"
            )
            return

        resolved = model_name
        if model_name == "bert-base-nli-mean-tokens":
            resolved = "sentence-transformers/bert-base-nli-mean-tokens"
        try:
            self._model = SentenceTransformer(resolved)
        except Exception as exc:
            print(
                f"[nsga3] failed to load SBERT {resolved!r}: {exc}\n"
                "        falling back to hash-based embedding"
            )

    # ------------------------------------------------------------------

    def embed(self, nodes: List[ClassNode]) -> np.ndarray:
        texts = [n.text_description or n.simple_name for n in nodes]
        if self._model is not None:
            vecs = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vecs.astype(np.float32)
        return np.stack([_hash_embed(t) for t in texts]).astype(np.float32)

    @staticmethod
    def cosine_matrix(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vecs / norms
        return unit @ unit.T


# ---------------------------------------------------------------------------


def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec
