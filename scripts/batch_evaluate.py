"""Run the compliance pipeline across every GDPR + HIPAA test document and
record per-document metrics.

For each doc:
 1. Run `backend.graph.run_pipeline`.
 2. Copy the resulting POA&M PDFs + raw violation_report to
    `outputs/POA&M/{doc_stem}/`.
 3. Compare predicted verdicts against the ground truth parsed from
    `data/testing/ground_truth/{reg}_annotations.json` and write
    `outputs/POA&M/{doc_stem}/metrics.json`.

Finally an aggregated summary is written to
`outputs/POA&M/_overall_summary.json`.

Usage:
    python scripts/batch_evaluate.py
    python scripts/batch_evaluate.py --regulations gdpr
    python scripts/batch_evaluate.py --limit 1      # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.metrics import compute_metrics  # noqa: E402

GDPR_FOCUS = ["art_5", "art_6", "art_7", "art_13", "art_14",
              "art_17", "art_25", "art_32", "art_33", "art_44"]
HIPAA_FOCUS = ["hipaa_164_306", "hipaa_164_308", "hipaa_164_310", "hipaa_164_312",
               "hipaa_164_316", "hipaa_164_404", "hipaa_164_408",
               "hipaa_164_502", "hipaa_164_524", "hipaa_164_530"]
FOCUS = {"gdpr": GDPR_FOCUS, "hipaa": HIPAA_FOCUS}


def _load_ground_truth(regulation: str) -> dict:
    path = PROJECT_ROOT / "data" / "testing" / "ground_truth" / f"{regulation}_annotations.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth missing at {path}. Run scripts/parse_annotations.py first."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _three_class_kappa(y_true: list[str], y_pred: list[str]) -> float:
    labels = ["Full", "Partial", "Missing"]
    n = len(y_true)
    if n == 0:
        return 0.0
    agree = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    po = agree / n
    pe = 0.0
    for lbl in labels:
        pt = sum(1 for t in y_true if t == lbl) / n
        pp = sum(1 for p in y_pred if p == lbl) / n
        pe += pt * pp
    return (po - pe) / (1 - pe) if (1 - pe) else 0.0


def _evaluate_doc(
    regulation: str,
    doc_stem: str,
    doc_path: str,
    gt_verdicts: dict[str, str],
) -> dict:
    """Run the pipeline on one document and return per-doc metrics."""
    from backend.graph import run_pipeline

    t0 = time.time()
    final_state = run_pipeline(doc_path, thinking=True)
    elapsed = time.time() - t0

    doc_id = final_state["doc_id"]
    report_src_dir = PROJECT_ROOT / "outputs" / "reports" / doc_id

    # Re-home reports into outputs/POA&M/{doc_stem}/
    dest_dir = PROJECT_ROOT / "outputs" / "POA&M" / doc_stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    poam_src = report_src_dir / "POA&M"
    if poam_src.exists():
        for name in ("assessment_report.pdf", "assessment_report.md",
                     "remediation_report.pdf", "remediation_report.md"):
            src = poam_src / name
            if src.exists():
                shutil.copy2(src, dest_dir / name)
    raw_src = report_src_dir / "raw" / "violation_report.json"
    if raw_src.exists():
        shutil.copy2(raw_src, dest_dir / "violation_report.json")

    vr = final_state.get("violation_report", {}) or {}
    violations = vr.get("violations", []) or []
    pred_map = {v["article_id"]: v["verdict"] for v in violations}

    focus = FOCUS[regulation]
    y_true = [gt_verdicts.get(a, "Full") for a in focus]
    y_pred = [pred_map.get(a, "Missing") for a in focus]

    # Binary P/R/F1/Kappa (Full vs not-Full) — matches backend.evaluation.metrics
    binary = compute_metrics(
        predictions={doc_stem: {a: p for a, p in zip(focus, y_pred)}},
        ground_truth={doc_stem: {a: {"label": t} for a, t in zip(focus, y_true)}},
    )

    three_class_kappa = round(_three_class_kappa(y_true, y_pred), 3)
    exact_match = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(focus)

    retrieved_clauses = final_state.get("retrieved_clauses", []) or []
    retrieved_article_ids: set[str] = set()
    for entry in retrieved_clauses:
        aid = entry.get("article_id") if isinstance(entry, dict) else None
        if aid:
            retrieved_article_ids.add(aid)
    focus_set = set(focus)
    retrieval_hit = focus_set & retrieved_article_ids
    retrieval = {
        "focus_articles": len(focus),
        "retrieved_distinct_articles": len(retrieved_article_ids),
        "focus_articles_retrieved": len(retrieval_hit),
        "recall_at_focus": round(len(retrieval_hit) / len(focus), 3) if focus else 0.0,
        "articles_debated": vr.get("articles_evaluated", 0),
        "debate_coverage": round(vr.get("articles_evaluated", 0) / len(focus), 3)
                           if focus else 0.0,
    }

    per_article = [
        {
            "article_id": a,
            "ground_truth": t,
            "predicted": p,
            "correct": t == p,
        }
        for a, t, p in zip(focus, y_true, y_pred)
    ]

    metrics = {
        "doc_stem": doc_stem,
        "doc_id": doc_id,
        "regulation": regulation,
        "elapsed_seconds": round(elapsed, 1),
        "risk_score": final_state.get("risk_score"),
        "risk_level": final_state.get("risk_level"),
        "hallucination_rate": vr.get("hallucination_rate", 0.0),
        "clause_evaluation": {
            "precision": binary["precision"],
            "recall": binary["recall"],
            "f1": binary["f1"],
            "cohens_kappa_binary": binary["cohens_kappa"],
            "cohens_kappa_3class": three_class_kappa,
            "exact_match_accuracy": round(exact_match, 3),
            "tp": binary["tp"], "fp": binary["fp"],
            "fn": binary["fn"], "tn": binary["tn"],
        },
        "retrieval_quality": retrieval,
        "per_article": per_article,
        "prediction_counts": {
            lbl: sum(1 for p in y_pred if p == lbl)
            for lbl in ("Full", "Partial", "Missing")
        },
        "ground_truth_counts": {
            lbl: sum(1 for t in y_true if t == lbl)
            for lbl in ("Full", "Partial", "Missing")
        },
    }

    with open(dest_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def _aggregate(all_metrics: list[dict]) -> dict:
    all_true: list[str] = []
    all_pred: list[str] = []
    for m in all_metrics:
        for row in m["per_article"]:
            all_true.append(row["ground_truth"])
            all_pred.append(row["predicted"])

    binary = compute_metrics(
        predictions={"_all": {str(i): p for i, p in enumerate(all_pred)}},
        ground_truth={"_all": {str(i): {"label": t} for i, t in enumerate(all_true)}},
    )
    three_class = round(_three_class_kappa(all_true, all_pred), 3)
    exact_match = round(sum(1 for t, p in zip(all_true, all_pred) if t == p) / len(all_true), 3) if all_true else 0.0

    def _mean(key_path: list[str]) -> float:
        vals = []
        for m in all_metrics:
            cur = m
            for k in key_path:
                cur = cur.get(k, {}) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "n_documents": len(all_metrics),
        "n_article_evaluations": len(all_true),
        "clause_evaluation_micro": {
            "precision": binary["precision"],
            "recall": binary["recall"],
            "f1": binary["f1"],
            "cohens_kappa_binary": binary["cohens_kappa"],
            "cohens_kappa_3class": three_class,
            "exact_match_accuracy": exact_match,
            "tp": binary["tp"], "fp": binary["fp"],
            "fn": binary["fn"], "tn": binary["tn"],
        },
        "retrieval_quality_mean": {
            "recall_at_focus": _mean(["retrieval_quality", "recall_at_focus"]),
            "debate_coverage": _mean(["retrieval_quality", "debate_coverage"]),
        },
        "hallucination_rate_mean": _mean(["hallucination_rate"]),
        "elapsed_seconds_total": round(sum(m["elapsed_seconds"] for m in all_metrics), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regulations", nargs="+", default=["gdpr", "hipaa"],
                        choices=["gdpr", "hipaa"])
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional cap on docs per regulation (0 = all).")
    args = parser.parse_args()

    (PROJECT_ROOT / "outputs" / "POA&M").mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    per_reg_summary: dict[str, dict] = {}

    for reg in args.regulations:
        gt_all = _load_ground_truth(reg)
        docs = list(gt_all.items())
        if args.limit:
            docs = docs[: args.limit]

        reg_metrics: list[dict] = []
        for idx, (doc_stem, payload) in enumerate(docs, 1):
            doc_path = payload.get("article_pdf")
            if not doc_path or not Path(doc_path).exists():
                print(f"[{reg}][{idx}/{len(docs)}] SKIP {doc_stem} — article pdf missing")
                continue
            print(f"[{reg}][{idx}/{len(docs)}] evaluating {doc_stem} ...", flush=True)
            try:
                metrics = _evaluate_doc(reg, doc_stem, doc_path, payload["verdicts"])
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            ce = metrics["clause_evaluation"]
            rq = metrics["retrieval_quality"]
            print(f"  done in {metrics['elapsed_seconds']}s — "
                  f"P={ce['precision']} R={ce['recall']} F1={ce['f1']} "
                  f"κ_bin={ce['cohens_kappa_binary']} κ_3={ce['cohens_kappa_3class']} "
                  f"exact={ce['exact_match_accuracy']} | "
                  f"retrieval_recall@focus={rq['recall_at_focus']}")
            reg_metrics.append(metrics)
            all_metrics.append(metrics)

        per_reg_summary[reg] = _aggregate(reg_metrics) if reg_metrics else {}

    overall = _aggregate(all_metrics) if all_metrics else {}
    summary_path = PROJECT_ROOT / "outputs" / "POA&M" / "_overall_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "per_regulation": per_reg_summary,
            "overall": overall,
            "docs": [m["doc_stem"] for m in all_metrics],
        }, fh, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
