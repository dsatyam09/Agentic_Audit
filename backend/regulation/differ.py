"""Regulation Differ — semantic comparison between old and new article versions.

Computes cosine similarity via the project's shared ``embedder`` singleton.
A similarity of 1.0 means the texts are semantically identical; values below
the configurable ``UPDATE_THRESHOLD`` (default 0.95) signal a meaningful change
that warrants re-indexing.

Usage:
    from backend.regulation.differ import semantic_diff
    sim = semantic_diff(old_article_text, new_article_text)
    if sim < 0.95:
        print("Regulation article has changed meaningfully.")
"""

from __future__ import annotations

import numpy as np

from backend.retrieval.embedder import embedder


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity in [0, 1] between two embedding vectors.

    A small epsilon is added to the denominator to avoid division-by-zero
    when either vector is all-zeros.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9
    return float(np.dot(a_arr, b_arr) / denom)


def semantic_diff(old_content: str, new_content: str) -> float:
    """Return the cosine similarity (0-1) between the embeddings of two texts.

    Parameters
    ----------
    old_content : str
        The previous version of the regulation article text.
    new_content : str
        The updated version of the regulation article text.

    Returns
    -------
    float
        Cosine similarity in ``[0, 1]``.  A value of 1.0 indicates the two
        texts are semantically identical.  Lower values indicate greater
        semantic divergence.

    Notes
    -----
    Both texts are embedded via the shared ``Embedder`` singleton (local
    ``all-MiniLM-L6-v2``), so repeated calls with the same text benefit
    from the in-memory SHA-256 cache.
    """
    if old_content == new_content:
        return 1.0

    if not old_content or not new_content:
        return 0.0

    vec_old = embedder.embed(old_content)
    vec_new = embedder.embed(new_content)
    return cosine_similarity(vec_old, vec_new)
