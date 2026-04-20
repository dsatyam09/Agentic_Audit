"""Post-hoc enrichment: recompute retrieval metrics from pipeline_logs.db.

Pre-guarantee retrieval recall = (n_focus - guaranteed_additions) / n_focus.
Also captures: total chunks, clauses retrieved, regulation versions.

Rewrites each outputs/POA&M/{doc_stem}/metrics.json and regenerates
_overall_summary.json.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB = PROJECT_ROOT / "outputs" / "pipeline_logs.db"
POAM_ROOT = PROJECT_ROOT / "outputs" / "POA&M"

GDPR_FOCUS = ["art_5", "art_6", "art_7", "art_13", "art_14",
              "art_17", "art_25", "art_32", "art_33", "art_44"]
HIPAA_FOCUS = ["hipaa_164_306", "hipaa_164_308", "hipaa_164_310", "hipaa_164_312",
               "hipaa_164_316", "hipaa_164_404", "hipaa_164_408",
               "hipaa_164_502", "hipaa_164_524", "hipaa_164_530"]
FOCUS = {"gdpr": GDPR_FOCUS, "hipaa": HIPAA_FOCUS}


def _latest_retrieval_log(doc_id: str) -> dict:
    """Return the most-recent retrieval log for a doc_id."""
    conn = sqlite3.connect(DB)
    try:
        cur = conn.execute(
            "SELECT structured_output FROM pipeline_logs "
            "WHERE doc_id = ? AND agent = 'retrieval' "
            "ORDER BY id DESC LIMIT 1",
            (doc_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


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


def _rebuild_doc_metrics(doc_dir: Path) -> dict:
    metrics_path = doc_dir / "metrics.json"
    with open(metrics_path) as fh:
        m = json.load(fh)

    regulation = m["regulation"]
    doc_id = m["doc_id"]
    focus = FOCUS[regulation]
    n_focus = len(focus)

    ret_log = _latest_retrieval_log(doc_id)
    guaranteed = int(ret_log.get("guaranteed_additions", 0))
    covered_articles = ret_log.get("covered_articles", []) or []
    covered_focus = [a for a in covered_articles if a in focus]

    # Classifier alignment: did the pipeline route this doc to the correct reg?
    vr_path = doc_dir / "violation_report.json"
    evaluated_against: list[str] = []
    if vr_path.exists():
        with open(vr_path) as fh:
            vr = json.load(fh)
        evaluated_against = vr.get("regulations", []) or []
    classifier_correct = regulation in evaluated_against
    m["classifier_alignment"] = {
        "expected_regulation": regulation,
        "evaluated_against": evaluated_against,
        "classifier_correct": classifier_correct,
    }

    pre_guarantee_hits = max(n_focus - guaranteed, 0)
    recall_at_focus_organic = round(pre_guarantee_hits / n_focus, 3) if n_focus else 0.0
    recall_at_focus_guaranteed = round(len(covered_focus) / n_focus, 3) if n_focus else 0.0

    m["retrieval_quality"] = {
        "focus_articles": n_focus,
        "focus_articles_retrieved_organic": pre_guarantee_hits,
        "focus_articles_retrieved_final": len(covered_focus),
        "guaranteed_additions": guaranteed,
        "recall_at_focus_organic": recall_at_focus_organic,
        "recall_at_focus_guaranteed": recall_at_focus_guaranteed,
        "articles_debated": m["retrieval_quality"].get("articles_debated", 0),
        "debate_coverage": m["retrieval_quality"].get("debate_coverage", 0.0),
        "total_clauses_retrieved": ret_log.get("total_clauses_retrieved", 0),
        "total_chunks": ret_log.get("total_chunks", 0),
    }

    with open(metrics_path, "w") as fh:
        json.dump(m, fh, indent=2)
    return m


def _aggregate(metrics_list: list[dict]) -> dict:
    y_true: list[str] = []
    y_pred: list[str] = []
    for m in metrics_list:
        for row in m["per_article"]:
            y_true.append(row["ground_truth"])
            y_pred.append(row["predicted"])

    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        gt_pos = t == "Full"
        pr_pos = p == "Full"
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
        k_bin = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    else:
        k_bin = 0.0

    def _mean(path: list[str]) -> float:
        vals: list[float] = []
        for m in metrics_list:
            cur = m
            for key in path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(key)
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "n_documents": len(metrics_list),
        "n_article_evaluations": len(y_true),
        "clause_evaluation_micro": {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "cohens_kappa_binary": round(k_bin, 3),
            "cohens_kappa_3class": round(_three_class_kappa(y_true, y_pred), 3),
            "exact_match_accuracy": round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true), 3) if y_true else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "retrieval_quality_mean": {
            "recall_at_focus_organic": _mean(["retrieval_quality", "recall_at_focus_organic"]),
            "recall_at_focus_guaranteed": _mean(["retrieval_quality", "recall_at_focus_guaranteed"]),
            "debate_coverage": _mean(["retrieval_quality", "debate_coverage"]),
            "guaranteed_additions_mean": _mean(["retrieval_quality", "guaranteed_additions"]),
        },
        "hallucination_rate_mean": _mean(["hallucination_rate"]),
        "elapsed_seconds_total": round(sum(m.get("elapsed_seconds", 0) for m in metrics_list), 1),
        "classifier_accuracy": round(
            sum(1 for m in metrics_list if m.get("classifier_alignment", {}).get("classifier_correct"))
            / len(metrics_list), 3
        ) if metrics_list else 0.0,
    }


def main():
    by_reg: dict[str, list[dict]] = {"gdpr": [], "hipaa": []}
    all_metrics: list[dict] = []

    for doc_dir in sorted(POAM_ROOT.iterdir()):
        if not doc_dir.is_dir():
            continue
        if not (doc_dir / "metrics.json").exists():
            continue
        m = _rebuild_doc_metrics(doc_dir)
        all_metrics.append(m)
        by_reg[m["regulation"]].append(m)

    summary = {
        "per_regulation": {r: _aggregate(v) for r, v in by_reg.items() if v},
        "overall": _aggregate(all_metrics),
        "docs": [m["doc_stem"] for m in all_metrics],
    }
    out = POAM_ROOT / "_overall_summary.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Rewrote {out}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
