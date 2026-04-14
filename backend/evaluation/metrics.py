"""Evaluation metrics: Precision, Recall, F1, Cohen's Kappa.

Binary classification: Full = compliant (positive), Partial/Missing = non-compliant (negative).
"""

from collections import Counter


def compute_metrics(predictions: dict, ground_truth: dict) -> dict:
    """
    Compute precision, recall, F1, and Cohen's kappa.

    Args:
        predictions: {doc_id: {article_id: verdict_label}}
        ground_truth: {doc_id: {article_id: {"label": ..., "notes": ...}}}

    Returns:
        Dict with precision, recall, f1, cohens_kappa, tp, fp, fn, tn
    """
    tp = fp = fn = tn = 0

    for doc_id in ground_truth:
        for art_id in ground_truth[doc_id]:
            gt = ground_truth[doc_id][art_id]
            gt_label = gt["label"] if isinstance(gt, dict) else gt
            pred = predictions.get(doc_id, {}).get(art_id, "Missing")

            gt_pos = (gt_label == "Full")
            pred_pos = (pred == "Full")

            if gt_pos and pred_pos:
                tp += 1
            elif pred_pos and not gt_pos:
                fp += 1
            elif gt_pos and not pred_pos:
                fn += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    # Cohen's Kappa
    if total > 0:
        po = (tp + tn) / total
        p_yes = ((tp + fp) / total) * ((tp + fn) / total)
        p_no = ((fn + tn) / total) * ((fp + tn) / total)
        pe = p_yes + p_no
        kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    else:
        kappa = 0.0

    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "cohens_kappa": round(kappa, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
    }


def compute_hallucination_rate(debate_records: list[dict]) -> dict:
    """Compute hallucination rate from debate records."""
    total = len(debate_records)
    flagged = sum(1 for r in debate_records if r.get("hallucination_flag", False))
    rate = flagged / total if total else 0.0
    return {
        "total_evaluations": total,
        "hallucination_flags": flagged,
        "hallucination_rate": round(rate, 4),
    }


def compute_debate_consistency(debate_records: list[dict], ground_truth: dict, doc_id: str) -> dict:
    """Compute % where Arbiter aligns with the correct side per ground truth."""
    total = 0
    aligned = 0

    gt_doc = ground_truth.get(doc_id, {})
    for record in debate_records:
        art_id = record["article_id"]
        if art_id not in gt_doc:
            continue

        gt_label = gt_doc[art_id]
        gt_label = gt_label["label"] if isinstance(gt_label, dict) else gt_label
        verdict = record["verdict"]

        total += 1
        if gt_label == verdict:
            aligned += 1

    consistency = aligned / total if total else 0.0
    return {
        "total_evaluated": total,
        "aligned": aligned,
        "consistency_rate": round(consistency, 3),
    }
