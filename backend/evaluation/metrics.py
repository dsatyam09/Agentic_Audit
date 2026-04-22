"""Evaluation metrics: Precision, Recall, F1, Cohen's Kappa.

Binary classification: Full = compliant (positive), Partial/Missing = non-compliant (negative).
Multiclass (Full/Partial/Missing): macro-F1, per-class P/R/F1, multi-class Cohen's kappa.
"""

from collections import Counter
from typing import Any


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


LABELS_THREE = ("Full", "Partial", "Missing")


def _normalize_label(lbl: Any) -> str:
    """Coerce any input into one of Full/Partial/Missing (case-insensitive)."""
    if isinstance(lbl, dict):
        lbl = lbl.get("label", "Missing")
    if not isinstance(lbl, str):
        return "Missing"
    s = lbl.strip().lower()
    if s.startswith("full"):
        return "Full"
    if s.startswith("part"):
        return "Partial"
    return "Missing"


def compute_multiclass_metrics(
    predictions: dict,
    ground_truth: dict,
    labels: tuple[str, ...] = LABELS_THREE,
) -> dict:
    """Per-class precision/recall/F1 + macro-F1 + multi-class Cohen's kappa.

    Args:
        predictions:   flat ``{article_id: label}`` (label is Full/Partial/Missing).
        ground_truth:  flat ``{article_id: label}`` *or* ``{article_id: {label: ...}}``.
        labels:        ordered tuple of class labels.

    Returns:
        Dict with ``macro_f1``, ``cohens_kappa``, ``accuracy``, ``support``,
        ``per_class`` (per-label precision/recall/f1/support) and the confusion matrix.
    """
    gt_map = {k: _normalize_label(v) for k, v in (ground_truth or {}).items()}
    pred_map = {k: _normalize_label(v) for k, v in (predictions or {}).items()}

    # Score only on comparable items (intersection). This avoids penalizing
    # non-evaluated articles as implicit "Missing", which can badly skew recall.
    overlap_ids = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    missing_in_pred = sorted(set(gt_map.keys()) - set(pred_map.keys()))
    extra_in_pred = sorted(set(pred_map.keys()) - set(gt_map.keys()))

    y_true: list[str] = [gt_map[i] for i in overlap_ids]
    y_pred: list[str] = [pred_map[i] for i in overlap_ids]

    if not y_true:
        return {
            "support": 0,  # number of overlapped article ids
            "macro_f1": None,
            "cohens_kappa": None,
            "accuracy": None,
            "per_class": {lbl: None for lbl in labels},
            "confusion_matrix": None,
            "labels": list(labels),
            "coverage": {
                "ground_truth_total": len(gt_map),
                "predictions_total": len(pred_map),
                "overlap_total": 0,
                "missing_in_predictions": len(missing_in_pred),
                "extra_predictions": len(extra_in_pred),
            },
            "missing_in_predictions": missing_in_pred,
            "extra_predictions": extra_in_pred,
        }

    try:
        from sklearn.metrics import (
            precision_recall_fscore_support,
            cohen_kappa_score,
            accuracy_score,
            confusion_matrix,
        )
    except ImportError:
        return {
            "support": len(y_true),
            "macro_f1": None,
            "cohens_kappa": None,
            "accuracy": None,
            "per_class": {lbl: None for lbl in labels},
            "confusion_matrix": None,
            "labels": list(labels),
            "error": "scikit-learn not installed",
            "coverage": {
                "ground_truth_total": len(gt_map),
                "predictions_total": len(pred_map),
                "overlap_total": len(overlap_ids),
                "missing_in_predictions": len(missing_in_pred),
                "extra_predictions": len(extra_in_pred),
            },
            "missing_in_predictions": missing_in_pred,
            "extra_predictions": extra_in_pred,
        }

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    macro_f1 = float(f.mean()) if len(f) else 0.0
    # Cohen's kappa is not meaningful in degenerate single-class truth/pred
    # distributions; return None instead of a misleading 0.0.
    true_unique = len(set(y_true))
    pred_unique = len(set(y_pred))
    if true_unique < 2 or pred_unique < 2:
        kappa = None
    else:
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=list(labels)))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()

    per_class = {
        labels[i]: {
            "precision": round(float(p[i]), 3),
            "recall": round(float(r[i]), 3),
            "f1": round(float(f[i]), 3),
            "support": int(s[i]),
        }
        for i in range(len(labels))
    }

    return {
        "support": len(y_true),
        "macro_f1": round(macro_f1, 3),
        "cohens_kappa": round(kappa, 3) if kappa is not None else None,
        "accuracy": round(acc, 3),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": list(labels),
        "coverage": {
            "ground_truth_total": len(gt_map),
            "predictions_total": len(pred_map),
            "overlap_total": len(overlap_ids),
            "missing_in_predictions": len(missing_in_pred),
            "extra_predictions": len(extra_in_pred),
        },
        "missing_in_predictions": missing_in_pred,
        "extra_predictions": extra_in_pred,
        "kappa_note": (
            "Undefined for single-class distributions"
            if kappa is None
            else None
        ),
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
