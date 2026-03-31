"""
Semantic embedding module — generates embeddings for methods and
pre-computes the pairwise cosine-similarity matrix.

Supports two backends (configured via ``backend`` parameter):

- ``"local"`` (default): Uses SBERT ``bert-base-nli-mean-tokens`` locally.
  Requires ``pip install sentence-transformers``.
- ``"api"``: Uses an OpenAI-compatible embedding API (text-embedding-3-small, etc.).
  Requires ``pip install openai`` and an API key.
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from .data_models import Method


class SemanticEmbedder:
    """
    Produces method embeddings and a pre-computed similarity matrix.
    Delegates to either a local SBERT model or an LLM embedding API.
    """

    def __init__(
        self,
        backend: str = "local",
        # local backend options
        model_name: str = "bert-base-nli-mean-tokens",
        # api backend options
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.backend = backend
        self._local_model = None
        self._api_embedder = None

        if backend == "api":
            if not api_key:
                raise ValueError("llm_api_key is required when backend='api'")
            from .llm_backend import LLMEmbedder
            self._api_embedder = LLMEmbedder(
                api_key=api_key, base_url=base_url, model=embedding_model,
            )
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(model_name)
            except ImportError:
                print("[WARN] sentence-transformers not installed — "
                      "falling back to hash-based simulation")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_methods(self, methods: List[Method]) -> Dict[str, np.ndarray]:
        """Return {method_id: embedding_vector} for every method."""
        texts = [self._build_text(m) for m in methods]
        vectors = self._embed_texts_internal(texts)
        return {m.id: vectors[i] for i, m in enumerate(methods)}

    def build_similarity_matrix(
        self,
        methods: List[Method],
        embeddings: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return an NxN cosine-similarity matrix (values in [-1, 1])."""
        ids = [m.id for m in methods]
        mat = np.stack([embeddings[mid] for mid in ids])  # (N, dim)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        return mat @ mat.T

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed arbitrary text strings (used for URI class-name grouping)."""
        return self._embed_texts_internal(texts)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _embed_texts_internal(self, texts: List[str]) -> np.ndarray:
        if self._api_embedder is not None:
            return self._api_embedder.embed_texts(texts)
        if self._local_model is not None:
            return self._local_model.encode(
                texts, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )
        return np.array([self._hash_embed(t) for t in texts])

    # ------------------------------------------------------------------
    # Text building (paper section III-A2)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text(method: Method) -> str:
        """Concatenate camelCase-split method name + parameter names + class name."""
        parts: List[str] = _split_camel(method.name)
        parts.extend(method.parameter_names)
        parts.extend(_split_camel(method.class_name))
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Fallback hash-based embedding (no external dependencies)
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
