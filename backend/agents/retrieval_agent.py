"""RetrievalAgent: RAG + cross-encoder reranking.

No LLM call. Pure vector search + reranking.
"""

from backend.retrieval.vector_store import retrieve_and_rerank, vector_store
from backend.agents.state import ComplianceState
from backend.logging.pipeline_log import make_log_entry


def retrieval_node(state: ComplianceState) -> dict:
    """LangGraph node: retrieves relevant regulation clauses for each document chunk."""
    doc_chunks = state["doc_chunks"]
    regulation_scope = state["regulation_scope"]
    retrieved_clauses = []

    for chunk in doc_chunks:
        chunk_results = {
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["chunk_text"],
            "clauses": [],
        }
        seen_articles = set()

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

            # Deduplicate across regulations by article_id
            for r in results:
                if r["article_id"] not in seen_articles:
                    chunk_results["clauses"].append(r)
                    seen_articles.add(r["article_id"])

        retrieved_clauses.append(chunk_results)

    # Build log entry
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
        },
    )

    return {
        "retrieved_clauses": retrieved_clauses,
        "pipeline_log": [log],
    }
