"""Local embedding wrapper using sentence-transformers all-MiniLM-L6-v2.

Zero API cost — runs entirely on CPU/GPU locally.

Usage:
    from backend.retrieval.embedder import embedder
    vector = embedder.embed("some compliance clause text")
"""

from __future__ import annotations

import hashlib

import backend.hf_setup  # noqa: F401 — load `.env` + HF auth before Hub access
from backend.hf_setup import hub_auth_token

from sentence_transformers import SentenceTransformer

_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Generates embeddings via a local sentence-transformers model with deterministic caching.

    The cache is keyed by the SHA-256 hex digest of the input text so
    repeated queries (common during retrieval + rerank) never hit the model
    twice for the same string.
    """

    def __init__(self, model: str = _MODEL) -> None:
        self._model_name = model
        self._model = SentenceTransformer(model, token=hub_auth_token())
        self._cache: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*, served from cache when possible."""
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        vector = self._model.encode(text, convert_to_numpy=True).tolist()

        self._cache[cache_key] = vector
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call (uncached items only).

        Results are individually cached so subsequent single-item lookups
        benefit from the batch call.
        """
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        results: list[list[float] | None] = [self._cache.get(k) for k in keys]

        # Identify texts that still need encoding
        missing_indices = [i for i, r in enumerate(results) if r is None]
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            vectors = self._model.encode(missing_texts, convert_to_numpy=True).tolist()
            for idx, vector in zip(missing_indices, vectors):
                self._cache[keys[idx]] = vector
                results[idx] = vector

        return results  # type: ignore[return-value]

    @property
    def cache_size(self) -> int:
        """Number of cached embeddings (useful for diagnostics)."""
        return len(self._cache)


# Module-level singleton -----------------------------------------------
embedder = Embedder()
