"""
Semantic embedding module — generates SBERT embeddings for methods and
pre-computes the pairwise cosine-similarity matrix used by the NSGA-III
clustering objectives and the URI-generation step.

Model: ``bert-base-nli-mean-tokens`` (as specified in the MONO2REST paper).
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from .data_models import Method


class SemanticEmbedder:
    """
    Wraps a SentenceTransformer model to produce method embeddings and
    a pre-computed similarity matrix.
    """

    def __init__(self, model_name: str = "bert-base-nli-mean-tokens"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("[WARN] sentence-transformers not installed — falling back to hash-based simulation")
            self.model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_methods(self, methods: List[Method]) -> Dict[str, np.ndarray]:
        """Return {method_id: embedding_vector} for every method."""
        texts = [self._build_text(m) for m in methods]
        if self.model is not None:
            vectors = self.model.encode(texts, show_progress_bar=True,
                                        convert_to_numpy=True, normalize_embeddings=True)
        else:
            vectors = np.array([self._hash_embed(t) for t in texts])
        return {m.id: vectors[i] for i, m in enumerate(methods)}

    def build_similarity_matrix(
        self,
        methods: List[Method],
        embeddings: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return an N×N cosine-similarity matrix (values in [-1, 1])."""
        ids = [m.id for m in methods]
        mat = np.stack([embeddings[mid] for mid in ids])  # (N, dim)
        # Normalise just in case
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        sim = mat @ mat.T  # cosine similarity
        return sim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed arbitrary text strings (used for URI class-name grouping)."""
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.array([self._hash_embed(t) for t in texts])

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ------------------------------------------------------------------
    # Text building (paper §III-A2)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text(method: Method) -> str:
        """Concatenate camelCase-split method name + parameter names + class name."""
        parts: List[str] = _split_camel(method.name)
        parts.extend(method.parameter_names)
        parts.extend(_split_camel(method.class_name))
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Fallback hash-based embedding (when sentence-transformers is absent)
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec


def _split_camel(name: str) -> List[str]:
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return [w.lower() for w in parts.split()]
