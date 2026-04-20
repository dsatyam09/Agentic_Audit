"""Compute a complete evaluation bundle per document.

Runs post-hoc over the artifacts already produced by scripts/batch_evaluate.py:
    outputs/POA&M/{doc_stem}/violation_report.json
    outputs/pipeline_logs.db (for retrieval + log completeness)

Adds three classes of metrics on top of what batch_evaluate.py recorded:

1. **RAGAS-style proxies** (no external LLM judge — uses MiniLM + the
   pipeline's own cite-then-verify flag):
     - faithfulness_proxy   = 1 − hallucination_rate
                              (hallucination_flag is set by the judge when
                               final_cited_text is not literally present in
                               the doc — this *is* the cite-then-verify check)
     - answer_relevance     = mean cos(embed(reasoning), embed(question))
                              question = "Does the policy comply with <reg> <art>?"
     - context_precision    = mean rerank_score of retrieved clauses, normalised
                              to [0,1] via 1/(1+exp(−x)). For focus articles
                              that were added by the guarantee pass (rerank_score=0)
                              we emit the sigmoid(0)=0.5 floor, matching how
                              RAGAS downweights lukewarm retrieval.

2. **Classification metrics from three defensible perspectives**:
     - Full-as-positive  (compliance-certification view)
     - Non-compliant-as-positive  (audit-finding view, more informative
                                   when the system skews toward Missing)
     - Macro-averaged 3-class  (label-balanced)

3. **System performance**:
     - end_to_end_latency_seconds
     - pipeline_log_completeness_full   (% of logged steps with raw prompt + raw
                                         response + structured output)
     - pipeline_log_completeness_structured  (% of steps with structured output)

Writes the enriched per-doc metrics back to the same metrics.json and a new
`outputs/POA&M/_evaluation_report.json` aggregating per-regulation + overall.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POAM_ROOT = PROJECT_ROOT / "outputs" / "POA&M"
DB = PROJECT_ROOT / "outputs" / "pipeline_logs.db"

LABELS = ["Full", "Partial", "Missing"]


# ---------------------------------------------------------------------------
# Embedder — reuse the MiniLM instance the pipeline already uses
# ---------------------------------------------------------------------------

def _cos(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _binary_prf_kappa(y_true: list[str], y_pred: list[str], positive: set[str]):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        gt_pos = t in positive
        pr_pos = p in positive
        if gt_pos and pr_pos: tp += 1
        elif pr_pos and not gt_pos: fp += 1
        elif gt_pos and not pr_pos: fn += 1
        else: tn += 1
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    if total:
        po = (tp + tn) / total
        p_yes = ((tp + fp) / total) * ((tp + fn) / total)
        p_no = ((fn + tn) / total) * ((fp + tn) / total)
        pe = p_yes + p_no
        kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    else:
        kappa = 0.0
    return {
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "cohens_kappa": round(kappa, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _macro_prf(y_true: list[str], y_pred: list[str]):
    per_class = {}
    macro_p = macro_r = macro_f = 0.0
    for lbl in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lbl] = {"precision": round(prec, 3), "recall": round(rec, 3),
                          "f1": round(f1, 3), "support": sum(1 for t in y_true if t == lbl)}
        macro_p += prec; macro_r += rec; macro_f += f1
    n = len(LABELS)
    return {
        "per_class": per_class,
        "macro_precision": round(macro_p / n, 3),
        "macro_recall": round(macro_r / n, 3),
        "macro_f1": round(macro_f / n, 3),
    }


def _kappa_3class(y_true: list[str], y_pred: list[str]) -> float:
    n = len(y_true)
    if n == 0: return 0.0
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    pe = 0.0
    for lbl in LABELS:
        pt = sum(1 for t in y_true if t == lbl) / n
        pp = sum(1 for p in y_pred if p == lbl) / n
        pe += pt * pp
    return round((po - pe) / (1 - pe), 3) if (1 - pe) else 0.0


# ---------------------------------------------------------------------------
# Faithfulness / relevance / context precision proxies
# ---------------------------------------------------------------------------

def _faithfulness_proxy(vr: dict) -> dict:
    """1 − hallucination_rate. Each debate record has hallucination_flag set
    when the Arbiter's final_cited_text is not verbatim in the source doc —
    which is exactly the cite-then-verify check RAGAS' faithfulness approximates.
    """
    violations = vr.get("violations", []) or []
    n = len(violations)
    flagged = sum(1 for v in violations if v.get("hallucination_flag"))
    faith = 1 - flagged / n if n else 0.0

    cited_nonempty = sum(1 for v in violations if (v.get("final_cited_text") or "").strip())
    return {
        "faithfulness_proxy": round(faith, 4),
        "hallucination_rate": round(flagged / n, 4) if n else 0.0,
        "records_evaluated": n,
        "records_with_citation": cited_nonempty,
    }


def _answer_relevance_proxy(vr: dict, embed) -> dict:
    violations = vr.get("violations", []) or []
    sims = []
    skipped = 0
    for v in violations:
        reasoning = (v.get("reasoning") or "").strip()
        if not reasoning:
            skipped += 1
            continue
        q = (f"Does the policy comply with {v.get('regulation','').upper()} "
             f"{v.get('article_id','')} — {v.get('article_title','')}?")
        try:
            sims.append(_cos(embed(q), embed(reasoning)))
        except Exception:
            skipped += 1
    if not sims:
        return {"answer_relevance_proxy": 0.0, "n_scored": 0, "n_skipped": skipped}
    return {
        "answer_relevance_proxy": round(sum(sims) / len(sims), 4),
        "n_scored": len(sims),
        "n_skipped": skipped,
    }


def _context_precision_proxy(vr: dict, ret_log: dict, focus: list[str], embed) -> dict:
    """RAGAS-style context precision:
    for each (question, retrieved_clause) measure semantic relevance of the
    clause to the question, then average over clauses per question.

    We only have clause_text for the articles that ended up as violation rows
    (final_cited_text is the policy clause, clause_text in the focus corpus is
    the regulatory clause). Approximate by scoring cos(embed(question),
    embed(article_clause_text)) for every focus article retrieved.
    """
    violations = vr.get("violations", []) or []
    clause_by_aid = {v["article_id"]: v.get("clause_text", "") for v in violations}

    covered_articles = ret_log.get("covered_articles", []) or []
    organic_hits = max(len(focus) - int(ret_log.get("guaranteed_additions", 0)), 0)

    # Relevance of each retrieved focus article's clause text to the regulation question
    reg = vr.get("regulations", [""])[0]
    scores = []
    for aid in covered_articles:
        if aid not in clause_by_aid:
            continue
        clause_text = clause_by_aid[aid]
        if not clause_text:
            continue
        q = f"Does the policy comply with {reg.upper()} {aid}?"
        try:
            scores.append(_cos(embed(q), embed(clause_text)))
        except Exception:
            continue
    mean_rel = sum(scores) / len(scores) if scores else 0.0

    # Hit rate of focus articles among the retrieved set
    focus_hits = sum(1 for a in covered_articles if a in focus)
    hit_precision = focus_hits / max(len(covered_articles), 1)

    return {
        "context_precision_proxy": round(mean_rel, 4),
        "focus_hit_rate_in_retrieved": round(hit_precision, 4),
        "organic_focus_recall": round(organic_hits / len(focus), 4) if focus else 0.0,
        "guaranteed_additions": int(ret_log.get("guaranteed_additions", 0)),
        "retrieved_clauses_total": int(ret_log.get("total_clauses_retrieved", 0)),
    }


# ---------------------------------------------------------------------------
# System performance from pipeline_logs.db
# ---------------------------------------------------------------------------

def _log_completeness(doc_id: str) -> dict:
    conn = sqlite3.connect(DB)
    try:
        row = conn.execute(
            "SELECT run_id FROM pipeline_logs WHERE doc_id=? ORDER BY id DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
        if not row:
            return {"steps": 0, "full_io_pct": 0.0, "structured_pct": 0.0}
        run_id = row[0]
        rows = conn.execute(
            "SELECT raw_prompt, raw_response, thinking_trace, structured_output "
            "FROM pipeline_logs WHERE run_id=?",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    total = len(rows)
    full_io = sum(1 for r in rows if r[0] and r[1] and r[3])
    struct = sum(1 for r in rows if r[3])
    return {
        "steps": total,
        "full_io_pct": round(full_io / total * 100, 1) if total else 0.0,
        "structured_pct": round(struct / total * 100, 1) if total else 0.0,
        "run_id": run_id,
    }


def _latest_retrieval_log(doc_id: str) -> dict:
    conn = sqlite3.connect(DB)
    try:
        row = conn.execute(
            "SELECT structured_output FROM pipeline_logs "
            "WHERE doc_id=? AND agent='retrieval' ORDER BY id DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Focus articles
# ---------------------------------------------------------------------------

GDPR_FOCUS = ["art_5", "art_6", "art_7", "art_13", "art_14",
              "art_17", "art_25", "art_32", "art_33", "art_44"]
HIPAA_FOCUS = ["hipaa_164_306", "hipaa_164_308", "hipaa_164_310", "hipaa_164_312",
               "hipaa_164_316", "hipaa_164_404", "hipaa_164_408",
               "hipaa_164_502", "hipaa_164_524", "hipaa_164_530"]
FOCUS = {"gdpr": GDPR_FOCUS, "hipaa": HIPAA_FOCUS}


# ---------------------------------------------------------------------------
# Per-doc enrichment
# ---------------------------------------------------------------------------

def _enrich(doc_dir: Path, embed) -> dict:
    metrics_path = doc_dir / "metrics.json"
    with open(metrics_path) as fh:
        m = json.load(fh)

    vr_path = doc_dir / "violation_report.json"
    with open(vr_path) as fh:
        vr = json.load(fh)

    regulation = m["regulation"]
    doc_id = m["doc_id"]
    focus = FOCUS[regulation]

    ret_log = _latest_retrieval_log(doc_id)

    # RAGAS-style proxies
    faith = _faithfulness_proxy(vr)
    rel = _answer_relevance_proxy(vr, embed)
    ctx = _context_precision_proxy(vr, ret_log, focus, embed)

    # Classification (all three perspectives)
    y_true = [row["ground_truth"] for row in m["per_article"]]
    y_pred = [row["predicted"] for row in m["per_article"]]

    full_pos = _binary_prf_kappa(y_true, y_pred, {"Full"})
    nonc_pos = _binary_prf_kappa(y_true, y_pred, {"Partial", "Missing"})
    macro = _macro_prf(y_true, y_pred)
    kappa_3 = _kappa_3class(y_true, y_pred)
    exact = round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true), 3) if y_true else 0.0

    # System perf
    logc = _log_completeness(doc_id)

    m["ragas_proxy"] = {
        "faithfulness": faith["faithfulness_proxy"],
        "answer_relevance": rel["answer_relevance_proxy"],
        "context_precision": ctx["context_precision_proxy"],
        "details": {
            "faithfulness": faith,
            "answer_relevance": rel,
            "context_precision": ctx,
        },
    }
    m["clause_evaluation_full_as_positive"] = full_pos
    m["clause_evaluation_noncompliant_as_positive"] = nonc_pos
    m["clause_evaluation_macro_3class"] = {
        **macro,
        "cohens_kappa_3class": kappa_3,
        "exact_match_accuracy": exact,
    }
    m["system_performance"] = {
        "end_to_end_latency_seconds": m.get("elapsed_seconds"),
        "pipeline_log_steps": logc["steps"],
        "pipeline_log_full_io_pct": logc["full_io_pct"],
        "pipeline_log_structured_pct": logc["structured_pct"],
    }

    with open(metrics_path, "w") as fh:
        json.dump(m, fh, indent=2)
    return m


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}

    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    for m in metrics_list:
        for row in m["per_article"]:
            y_true_all.append(row["ground_truth"])
            y_pred_all.append(row["predicted"])

    def _mean(path: list[str]) -> float:
        vals: list[float] = []
        for m in metrics_list:
            cur = m
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None: break
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    full_pos = _binary_prf_kappa(y_true_all, y_pred_all, {"Full"})
    nonc_pos = _binary_prf_kappa(y_true_all, y_pred_all, {"Partial", "Missing"})
    macro = _macro_prf(y_true_all, y_pred_all)
    kappa_3 = _kappa_3class(y_true_all, y_pred_all)
    exact = round(sum(1 for t, p in zip(y_true_all, y_pred_all) if t == p) / len(y_true_all), 3) if y_true_all else 0.0

    return {
        "n_documents": len(metrics_list),
        "n_article_evaluations": len(y_true_all),
        "ragas_proxy_mean": {
            "faithfulness": _mean(["ragas_proxy", "faithfulness"]),
            "answer_relevance": _mean(["ragas_proxy", "answer_relevance"]),
            "context_precision": _mean(["ragas_proxy", "context_precision"]),
        },
        "clause_evaluation_full_as_positive": full_pos,
        "clause_evaluation_noncompliant_as_positive": nonc_pos,
        "clause_evaluation_macro_3class": {
            **macro,
            "cohens_kappa_3class": kappa_3,
            "exact_match_accuracy": exact,
        },
        "retrieval_quality_mean": {
            "recall_at_focus_organic": _mean(["retrieval_quality", "recall_at_focus_organic"]),
            "recall_at_focus_final": _mean(["retrieval_quality", "recall_at_focus_guaranteed"]),
            "guaranteed_additions_mean": _mean(["retrieval_quality", "guaranteed_additions"]),
        },
        "system_performance_mean": {
            "end_to_end_latency_seconds": _mean(["system_performance", "end_to_end_latency_seconds"]),
            "pipeline_log_full_io_pct": _mean(["system_performance", "pipeline_log_full_io_pct"]),
            "pipeline_log_structured_pct": _mean(["system_performance", "pipeline_log_structured_pct"]),
        },
        "hallucination_rate_mean": _mean(["hallucination_rate"]),
        "classifier_accuracy": round(
            sum(1 for m in metrics_list if m.get("classifier_alignment", {}).get("classifier_correct"))
            / len(metrics_list), 3
        ) if metrics_list else 0.0,
    }


def main():
    from backend.retrieval.embedder import embedder

    def embed(txt: str):
        return embedder.embed(txt)

    by_reg: dict[str, list[dict]] = {"gdpr": [], "hipaa": []}
    all_metrics: list[dict] = []
    for doc_dir in sorted(POAM_ROOT.iterdir()):
        if not doc_dir.is_dir() or not (doc_dir / "metrics.json").exists():
            continue
        m = _enrich(doc_dir, embed)
        all_metrics.append(m)
        by_reg[m["regulation"]].append(m)
        ce = m["clause_evaluation_macro_3class"]
        rg = m["ragas_proxy"]
        print(f"{m['doc_stem']}: faith={rg['faithfulness']} rel={rg['answer_relevance']} "
              f"ctx={rg['context_precision']} macroF1={ce['macro_f1']} "
              f"k3={ce['cohens_kappa_3class']} exact={ce['exact_match_accuracy']}")

    report = {
        "per_regulation": {r: _aggregate(ms) for r, ms in by_reg.items() if ms},
        "overall": _aggregate(all_metrics),
        "docs": [m["doc_stem"] for m in all_metrics],
    }
    out = POAM_ROOT / "_evaluation_report.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
