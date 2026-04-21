"""Cross-encoder reranker using sentence-transformers.

Uses the lightweight ``cross-encoder/ms-marco-MiniLM-L-6-v2`` model to
score (query, passage) pairs for semantic relevance.  The model is loaded
lazily on first call so import time stays fast.

Usage:
    from backend.retrieval.reranker import reranker
    scores = reranker.predict([("query", "passage A"), ("query", "passage B")])
"""

from __future__ import annotations

import backend.hf_setup  # noqa: F401 — load `.env` + HF auth before Hub access
from backend.hf_setup import hub_auth_token

from sentence_transformers import CrossEncoder

_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Wrapper around a sentence-transformers CrossEncoder for reranking."""

    def __init__(self, model_name: str = _MODEL) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name, token=hub_auth_token())
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return relevance scores for a list of (query, passage) pairs.

        Higher scores indicate stronger semantic relevance.  The scores
        are *not* normalised — they are raw logits from the cross-encoder.
        """
        model = self._load_model()
        scores = model.predict(pairs)
        return [float(s) for s in scores]


# Module-level singleton -----------------------------------------------
reranker = Reranker()
