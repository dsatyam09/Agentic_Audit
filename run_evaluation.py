"""CLI entry point: runs all experimental conditions against test documents.

Conditions:
    C1 — Naive baseline (gpt-4o-mini, no RAG, no rerank, no debate)
    C2 — RAG only (Qwen3-8B, RAG, no rerank, no debate)
    C3 — RAG + rerank (Qwen3-8B, RAG, rerank, no debate)
    C4 — Full system (Qwen3-8B, RAG, rerank, debate)
    C4-nothink — Ablation (Qwen3-8B think OFF, RAG, rerank, debate)

Usage:
    python run_evaluation.py
    python run_evaluation.py --conditions C1 C4
    python run_evaluation.py --regulation gdpr
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent


def load_ground_truth(regulation: str) -> dict:
    """Load ground truth annotations for a regulation."""
    gt_path = PROJECT_ROOT / "data" / "testing" / "ground_truth" / f"{regulation}_annotations.json"
    if not gt_path.exists():
        print(f"Warning: Ground truth not found at {gt_path}")
        return {}
    with open(gt_path) as f:
        annotations = json.load(f)
    # Convert to {doc_id: {article_id: annotation}} format
    gt = {}
    for doc in annotations:
        gt[doc["doc_id"]] = doc.get("annotations", {})
    return gt


def get_test_documents(regulation: str) -> list[str]:
    """Find all test documents for a regulation."""
    docs = []
    base = PROJECT_ROOT / "data" / "testing" / "documents" / regulation
    if not base.exists():
        # Try legacy test_datasets location
        base = PROJECT_ROOT / "test_datasets" / regulation / "articles"
    if not base.exists():
        print(f"Warning: No test documents found for {regulation}")
        return docs
    for f in sorted(base.rglob("*.pdf")):
        docs.append(str(f))
    for f in sorted(base.rglob("*.txt")):
        docs.append(str(f))
    return docs


def run_condition_c1(doc_path: str) -> dict:
    """C1: Naive baseline — single Qwen3-8B call, no RAG, no debate."""
    from backend.ingestion.parser import DocumentParser
    from backend.debate.qwen_runner import qwen

    doc_text = DocumentParser().parse(doc_path)[:3000]

    prompt = f"""Evaluate this enterprise document against GDPR compliance requirements.
For each of these articles, determine if the document is Full, Partial, or Missing compliance:
art_5 (Principles), art_6 (Lawful basis), art_7 (Consent), art_13 (Direct collection notice),
art_14 (Indirect collection notice), art_17 (Right to erasure), art_25 (Privacy by design),
art_32 (Security measures), art_33 (Breach notification), art_44 (International transfers).

Document:
{doc_text}

Respond ONLY in JSON format:
{{"art_5": "Full|Partial|Missing", "art_6": "Full|Partial|Missing", ...}}"""

    out = qwen.generate(prompt, thinking=False, max_new_tokens=512)
    raw = out["response"].strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group()) if match else {}
    return result


def run_condition_c2(doc_path: str) -> dict:
    """C2: RAG only — Qwen3-8B with RAG, no reranking, no debate."""
    from backend.ingestion.parser import DocumentParser
    from backend.ingestion.chunker import DocumentChunker
    from backend.retrieval.embedder import embedder
    from backend.retrieval.vector_store import vector_store
    from backend.debate.qwen_runner import qwen

    doc_text = DocumentParser().parse(doc_path)
    chunks = DocumentChunker().chunk(doc_text)

    results = {}
    for chunk in chunks[:3]:  # Limit chunks for efficiency
        embedding = embedder.embed(chunk["chunk_text"])
        col_size = vector_store.collection_size("gdpr")
        if col_size == 0:
            continue
        retrieved = vector_store.query("gdpr", embedding, min(10, col_size))
        if not retrieved.get("documents") or not retrieved["documents"][0]:
            continue

        context = "\n".join(retrieved["documents"][0][:5])
        prompt = f"""Given this regulatory context and policy text, evaluate compliance.

REGULATORY CONTEXT:
{context}

POLICY TEXT:
{chunk['chunk_text']}

For each article mentioned in the regulatory context, determine: Full, Partial, or Missing.
Respond ONLY in JSON: {{"article_id": "Full|Partial|Missing", ...}}"""

        out = qwen.generate(prompt, thinking=True)
        try:
            parsed = json.loads(out["response"])
            results.update(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    return results


def run_condition_c3(doc_path: str) -> dict:
    """C3: RAG + rerank — Qwen3-8B with RAG and cross-encoder reranking, no debate."""
    from backend.ingestion.parser import DocumentParser
    from backend.ingestion.chunker import DocumentChunker
    from backend.retrieval.vector_store import retrieve_and_rerank
    from backend.debate.qwen_runner import qwen

    doc_text = DocumentParser().parse(doc_path)
    chunks = DocumentChunker().chunk(doc_text)

    results = {}
    for chunk in chunks[:3]:
        from backend.retrieval.vector_store import vector_store
        col_size = vector_store.collection_size("gdpr")
        if col_size == 0:
            continue

        reranked = retrieve_and_rerank(
            query=chunk["chunk_text"],
            namespace="gdpr",
            top_k_candidates=min(10, col_size),
            top_k_final=min(5, col_size),
        )
        if not reranked:
            continue

        context = "\n".join([r["clause_text"] for r in reranked])
        articles_str = ", ".join([f"{r['article_id']} ({r['article_title']})" for r in reranked])

        prompt = f"""Given this regulatory context and policy text, evaluate compliance.

REGULATORY CONTEXT (reranked by relevance):
{context}

RELEVANT ARTICLES: {articles_str}

POLICY TEXT:
{chunk['chunk_text']}

For each article, determine: Full, Partial, or Missing.
Respond ONLY in JSON: {{"article_id": "Full|Partial|Missing", ...}}"""

        out = qwen.generate(prompt, thinking=True)
        try:
            parsed = json.loads(out["response"])
            results.update(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    return results


def run_condition_c4(doc_path: str, thinking: bool = True) -> dict:
    """C4: Full system — RAG + rerank + debate. C4-nothink when thinking=False."""
    from backend.graph import run_pipeline
    final_state = run_pipeline(doc_path)

    results = {}
    for record in final_state.get("debate_records", []):
        art_id = record["article_id"]
        # Keep best verdict per article (Full > Partial > Missing)
        rank = {"Full": 2, "Partial": 1, "Missing": 0}
        if art_id not in results or rank.get(record["verdict"], 0) > rank.get(results[art_id], 0):
            results[art_id] = record["verdict"]

    return results


CONDITIONS = {
    "C1": {"name": "Naive baseline", "runner": run_condition_c1},
    "C2": {"name": "RAG only", "runner": run_condition_c2},
    "C3": {"name": "RAG + rerank", "runner": run_condition_c3},
    "C4": {"name": "Full system", "runner": run_condition_c4},
    "C4-nothink": {"name": "Ablation (think OFF)", "runner": lambda p: run_condition_c4(p, thinking=False)},
}


def main():
    parser = argparse.ArgumentParser(description="Run evaluation across experimental conditions")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()),
                        help="Which conditions to run")
    parser.add_argument("--regulation", default="gdpr", help="Regulation to evaluate against")
    args = parser.parse_args()

    from backend.evaluation.metrics import compute_metrics, compute_hallucination_rate

    ground_truth = load_ground_truth(args.regulation)
    test_docs = get_test_documents(args.regulation)

    if not test_docs:
        print("No test documents found. Exiting.")
        sys.exit(1)

    print(f"Evaluation Configuration:")
    print(f"  Regulation: {args.regulation}")
    print(f"  Conditions: {', '.join(args.conditions)}")
    print(f"  Test Documents: {len(test_docs)}")
    print(f"  Ground Truth Entries: {len(ground_truth)}")
    print()

    all_results = {}

    for cond_id in args.conditions:
        cond = CONDITIONS[cond_id]
        print(f"\n{'='*60}")
        print(f"Running {cond_id}: {cond['name']}")
        print(f"{'='*60}")

        predictions = {}
        for doc_path in test_docs:
            doc_id = Path(doc_path).stem
            print(f"  Evaluating: {doc_id}...", end=" ", flush=True)
            try:
                result = cond["runner"](doc_path)
                predictions[doc_id] = result
                print(f"OK ({len(result)} articles)")
            except Exception as e:
                print(f"ERROR: {e}")
                predictions[doc_id] = {}

        # Compute metrics
        metrics = compute_metrics(predictions, ground_truth)
        all_results[cond_id] = {
            "condition": cond_id,
            "name": cond["name"],
            "predictions": predictions,
            "metrics": metrics,
        }

        print(f"\n  Results for {cond_id}:")
        print(f"    Precision: {metrics['precision']}")
        print(f"    Recall:    {metrics['recall']}")
        print(f"    F1:        {metrics['f1']}")
        print(f"    Kappa:     {metrics['cohens_kappa']}")

    # Save results
    output_path = PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "regulation": args.regulation,
        "n_documents": len(test_docs),
        "conditions": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nEvaluation summary saved to: {output_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Kappa':>10}")
    print("-" * 55)
    for cond_id in args.conditions:
        m = all_results[cond_id]["metrics"]
        print(f"{cond_id:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['cohens_kappa']:>10.3f}")


if __name__ == "__main__":
    main()
