"""ChromaDB-backed vector store with persistent storage and a convenience
``retrieve_and_rerank`` pipeline that ties together embedding, vector
search, and cross-encoder reranking.

Usage:
    from backend.retrieval.vector_store import vector_store, retrieve_and_rerank

    # Low-level
    vector_store.upsert("gdpr", ids, docs, embeddings, metadatas)
    results = vector_store.query("gdpr", query_embedding, n_results=20)

    # High-level convenience
    top = retrieve_and_rerank("Does the policy address data erasure?",
                              namespace="gdpr",
                              top_k_candidates=20,
                              top_k_final=5)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

# ---------------------------------------------------------------------------
# Resolve the persistent storage directory.
# ``data/chroma_db/`` is relative to the repository root, which we locate by
# walking upward from this file until we find the ``data/`` directory.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent  # backend/retrieval -> backend -> repo root
_DEFAULT_PERSIST_DIR = str(_REPO_ROOT / "data" / "chroma_db")


class VectorStore:
    """Thin wrapper around ChromaDB that organises embeddings by namespace.

    Each *namespace* (e.g. ``"gdpr"``, ``"hipaa"``, ``"nist"``) maps to a
    separate ChromaDB collection, keeping regulation data cleanly isolated.
    """

    def __init__(self, persist_directory: str | None = None) -> None:
        self._persist_dir = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", _DEFAULT_PERSIST_DIR
        )
        self._client = chromadb.PersistentClient(path=self._persist_dir)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_or_create_collection(self, namespace: str) -> Collection:
        """Return the ChromaDB collection for *namespace*, creating it if needed."""
        return self._client.get_or_create_collection(
            name=namespace,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        namespace: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update documents in the given namespace.

        All four parallel lists must have the same length.
        """
        collection = self.get_or_create_collection(namespace)
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        namespace: str,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve the *n_results* nearest neighbours from *namespace*.

        Returns the raw ChromaDB result dict with keys ``ids``,
        ``documents``, ``metadatas``, ``distances``, etc.
        """
        collection = self.get_or_create_collection(namespace)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, collection.count() or n_results),
        }
        if where:
            kwargs["where"] = where
        # Guard against querying an empty collection.
        if collection.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return collection.query(**kwargs)

    def collection_size(self, namespace: str) -> int:
        """Return the number of documents stored in *namespace*."""
        collection = self.get_or_create_collection(namespace)
        return collection.count()


# Module-level singleton -----------------------------------------------
vector_store = VectorStore()


# ======================================================================
# High-level retrieve-and-rerank pipeline
# ======================================================================

def retrieve_and_rerank(
    query: str,
    namespace: str,
    top_k_candidates: int = 20,
    top_k_final: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """End-to-end retrieval pipeline: embed → vector search → cross-encoder rerank.

    Parameters
    ----------
    query:
        Natural-language compliance question or policy excerpt.
    namespace:
        ChromaDB collection to search (e.g. ``"gdpr"``).
    top_k_candidates:
        Number of candidates to pull from the vector store (ANN phase).
    top_k_final:
        Number of results to return after cross-encoder reranking.
    where:
        Optional ChromaDB metadata filter dict.

    Returns
    -------
    list[dict]
        Sorted (highest relevance first) list of dicts, each containing:
        ``article_id``, ``article_title``, ``clause_text``, ``severity``,
        ``regulation``, ``rerank_score``.
    """
    # Lazy imports so the module can be imported without side-effects from
    # sibling singletons (useful for testing / mocking).
    from backend.retrieval.embedder import embedder
    from backend.retrieval.reranker import reranker

    # 1. Embed the query (cache-friendly via SHA-256 key).
    query_embedding = embedder.embed(query)

    # 2. ANN retrieval from ChromaDB.
    raw = vector_store.query(
        namespace=namespace,
        query_embedding=query_embedding,
        n_results=top_k_candidates,
        where=where,
    )

    ids: list[str] = raw["ids"][0]
    documents: list[str] = raw["documents"][0]
    metadatas: list[dict[str, Any]] = raw["metadatas"][0]

    if not ids:
        return []

    # 3. Cross-encoder reranking.
    pairs: list[tuple[str, str]] = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)

    # 4. Merge and sort.
    candidates: list[dict[str, Any]] = []
    for doc_id, doc_text, meta, score in zip(ids, documents, metadatas, scores):
        candidates.append(
            {
                "article_id": meta.get("article_id", doc_id),
                "article_title": meta.get("article_title", ""),
                "clause_text": doc_text,
                "severity": meta.get("severity", ""),
                "regulation": meta.get("regulation", namespace),
                "rerank_score": score,
            }
        )

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k_final]
