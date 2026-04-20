"""RetrievalAgent: RAG + cross-encoder reranking.

For every policy chunk, retrieves and reranks the top-k regulation clauses.
After the per-chunk pass, performs a **guarantee pass**: every focus article
in the active regulation scope must be debated at least once in the run, so
the final Assessment Report reflects the full focus article set rather than
only the articles that happened to rank into a chunk's top-k.

No LLM call. Pure vector search + reranking.
"""

from __future__ import annotations

from backend.agents.classifier import REGULATION_REGISTRY
from backend.agents.state import ComplianceState
from backend.logging.pipeline_log import make_log_entry
from backend.retrieval.vector_store import retrieve_and_rerank, vector_store


def _load_focus_article_clauses(regulation: str) -> dict[str, dict]:
    """Return a mapping of focus_article_id → minimal clause dict ready for debate.

    Reads the indexed corpus from Chroma so the clause_text matches what the
    debate will see as retrieval context.
    """
    from chromadb.errors import ChromaError  # type: ignore

    reg_cfg = REGULATION_REGISTRY.get(regulation, {})
    focus_ids: list[str] = reg_cfg.get("focus_articles", [])
    if not focus_ids:
        return {}

    collection = vector_store.get_or_create_collection(regulation)
    try:
        result = collection.get(ids=focus_ids, include=["documents", "metadatas"])
    except (ChromaError, ValueError):
        return {}

    clauses: dict[str, dict] = {}
    for aid, doc_text, meta in zip(
        result.get("ids", []),
        result.get("documents", []),
        result.get("metadatas", []),
    ):
        if not aid:
            continue
        clauses[aid] = {
            "article_id": meta.get("article_id", aid),
            "article_title": meta.get("article_title", ""),
            "clause_text": doc_text or "",
            "severity": meta.get("severity", "High"),
            "regulation": regulation,
            "rerank_score": 0.0,
        }
    return clauses


def _best_chunk_for_article(
    doc_chunks: list[dict],
    clause_text: str,
    top_k_candidates: int = 1,
) -> int:
    """Return the chunk_index that best matches *clause_text* by embedding similarity."""
    if not doc_chunks:
        return 0

    from backend.retrieval.embedder import embedder
    import numpy as np

    q = np.asarray(embedder.embed(clause_text), dtype=np.float64)
    q_norm = np.linalg.norm(q) + 1e-9

    best_idx = 0
    best_sim = -1.0
    for chunk in doc_chunks:
        c = np.asarray(embedder.embed(chunk["chunk_text"]), dtype=np.float64)
        c_norm = np.linalg.norm(c) + 1e-9
        sim = float(np.dot(q, c) / (q_norm * c_norm))
        if sim > best_sim:
            best_sim = sim
            best_idx = chunk["chunk_index"]
    return best_idx


def retrieval_node(state: ComplianceState) -> dict:
    """LangGraph node: retrieves relevant regulation clauses for each document chunk.

    Also performs a guarantee-pass so every regulation focus article is paired
    with at least one chunk for debate.
    """
    doc_chunks = state["doc_chunks"]
    regulation_scope = state["regulation_scope"]
    retrieved_clauses: list[dict] = []

    # Track (chunk_index -> set of article_ids already attached to that chunk)
    # so the guarantee pass doesn't duplicate clauses already present.
    per_chunk_articles: dict[int, set[str]] = {}
    # Global set of articles covered anywhere in the run.
    covered_articles: set[str] = set()

    chunk_by_index: dict[int, dict] = {}

    for chunk in doc_chunks:
        idx = chunk["chunk_index"]
        chunk_results = {
            "chunk_index": idx,
            "chunk_text": chunk["chunk_text"],
            "clauses": [],
        }
        seen_articles: set[str] = set()

        for regulation in regulation_scope:
            col_size = vector_store.collection_size(regulation)
            if col_size == 0:
                continue

            top_k_candidates = min(10, col_size)
            top_k_final = min(5, col_size)

            results = retrieve_and_rerank(
                query=chunk["chunk_text"],
                namespace=regulation,
                top_k_candidates=top_k_candidates,
                top_k_final=top_k_final,
            )

            for r in results:
                if r["article_id"] not in seen_articles:
                    chunk_results["clauses"].append(r)
                    seen_articles.add(r["article_id"])
                    covered_articles.add(r["article_id"])

        per_chunk_articles[idx] = seen_articles
        chunk_by_index[idx] = chunk_results
        retrieved_clauses.append(chunk_results)

    # ── Guarantee pass ─────────────────────────────────────────────────
    # For every focus article that did not appear in any chunk's top-k,
    # pair it with the best-matching chunk so it gets debated at least once.
    guaranteed_additions = 0
    for regulation in regulation_scope:
        focus_clauses = _load_focus_article_clauses(regulation)
        for aid, clause in focus_clauses.items():
            if aid in covered_articles:
                continue
            if not doc_chunks:
                break
            best_idx = _best_chunk_for_article(doc_chunks, clause["clause_text"])
            target = chunk_by_index.get(best_idx)
            if target is None:
                continue
            if aid in per_chunk_articles.setdefault(best_idx, set()):
                continue
            target["clauses"].append(clause)
            per_chunk_articles[best_idx].add(aid)
            covered_articles.add(aid)
            guaranteed_additions += 1

    total_clauses = sum(len(c["clauses"]) for c in retrieved_clauses)
    log = make_log_entry(
        agent="retrieval",
        input_data={
            "n_chunks": len(doc_chunks),
            "regulations": regulation_scope,
        },
        raw_prompt=None,
        thinking_trace=None,
        raw_response=None,
        structured_output={
            "total_chunks": len(doc_chunks),
            "total_clauses_retrieved": total_clauses,
            "regulations_searched": regulation_scope,
            "guaranteed_additions": guaranteed_additions,
            "covered_articles": sorted(covered_articles),
        },
    )

    return {
        "retrieved_clauses": retrieved_clauses,
        "pipeline_log": [log],
    }
