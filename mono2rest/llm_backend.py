"""
LLM API backend — provides embedding and HTTP-method classification via
OpenAI-compatible API endpoints.

Supports any provider that follows the OpenAI API format:
  - OpenAI (api.openai.com)
  - DeepSeek, Moonshot, Zhipu, SiliconFlow, etc.
  - Local servers (Ollama, vLLM, LM Studio …)

Usage::

    config = {
        "llm_backend": "api",
        "llm_api_key": "sk-...",
        "llm_base_url": "https://api.openai.com/v1",    # optional
        "llm_embedding_model": "text-embedding-3-small", # optional
        "llm_chat_model": "gpt-4o-mini",                 # optional
    }
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

import numpy as np

from .data_models import HTTPMethod, Method


class LLMEmbedder:
    """Generate embeddings via an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts and return an (N, dim) numpy array."""
        # OpenAI embedding API supports batches up to ~2048 items
        batch_size = 512
        all_vecs: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            resp = self.client.embeddings.create(input=batch, model=self.model)
            vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            all_vecs.extend(vecs)
        mat = np.stack(all_vecs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class LLMHttpClassifier:
    """Classify Java method signatures into HTTP verbs via chat completion."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def classify_batch(self, methods: List[Method]) -> List[HTTPMethod]:
        """Classify multiple method signatures in a single LLM call."""
        if not methods:
            return []

        # Build batch prompt
        sigs = []
        for i, m in enumerate(methods):
            sigs.append(f"{i+1}. {m.get_signature_text()}")
        sigs_text = "\n".join(sigs)

        prompt = (
            "You are classifying Java method signatures into REST HTTP methods.\n"
            "For each method below, reply with ONLY a JSON array of objects "
            "like [{\"id\":1,\"http\":\"GET\"}, ...]\n"
            "Choose from: GET, POST, PUT, DELETE.\n"
            "Rules:\n"
            "- Methods that retrieve/find/query data → GET\n"
            "- Methods that create/add/insert/save new data → POST\n"
            "- Methods that update/modify/edit existing data → PUT\n"
            "- Methods that delete/remove data → DELETE\n\n"
            f"Methods:\n{sigs_text}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            # Extract JSON from response (may be wrapped in markdown code block)
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                result_map: Dict[int, str] = {}
                for item in items:
                    result_map[item["id"]] = item["http"].upper()
                results = []
                for i in range(len(methods)):
                    verb = result_map.get(i + 1, "GET")
                    results.append(HTTPMethod(verb) if verb in ("GET", "POST", "PUT", "DELETE")
                                   else HTTPMethod.GET)
                return results
            else:
                raise ValueError(f"No JSON array found in LLM response: {text[:200]}")
        except Exception as e:
            print(f"[WARN] LLM classification failed: {e}, falling back to heuristic")
            from .rest_api_generator import RESTAPIGenerator
            return [RESTAPIGenerator._classify_heuristic(m) for m in methods]
