"""Compliance Drift Detector — compares two ViolationReports and computes
a Semantic Regression Score (SRS) for every regression.

SRS formula:
    SRS = (coverage_rank_drop x risk_weight) x (1 + cosine_distance(cited_v1, cited_v2))

A clause removal scores SRS ~ 4.0.  A subtle language weakening scores ~ 1.5.
This prioritises remediation effort by combining structural coverage loss with
semantic divergence.

Usage:
    from backend.drift.detector import detect_drift, drift_node
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from backend.retrieval.embedder import embedder

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
RISK_WEIGHTS: dict[str, int] = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
COVERAGE_RANK: dict[str, int] = {"Full": 2, "Partial": 1, "Missing": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_distance(a: list[float], b: list[float]) -> float:
    """Return 1 - cosine_similarity, clamped to [0, 1].

    A small epsilon is added to the denominator to avoid division by zero when
    one of the vectors is all-zeros (e.g. empty-text embedding edge case).
    """
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9
    return float(1.0 - np.dot(a_arr, b_arr) / denom)


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def detect_drift(report_v1: dict, report_v2: dict) -> dict:
    """Compare two ViolationReports and compute drift with SRS scores.

    Parameters
    ----------
    report_v1 : dict
        The *older* violation report (baseline).
    report_v2 : dict
        The *newer* violation report.

    Returns
    -------
    dict
        A DriftResult containing regressions (sorted by SRS desc),
        improvements, aggregate risk-score delta, and critical-regression IDs.
    """
    v1 = {v["article_id"]: v for v in report_v1["violations"]}
    v2 = {v["article_id"]: v for v in report_v2["violations"]}

    regressions: list[dict] = []
    improvements: list[dict] = []

    for art_id in set(v1) | set(v2):
        v1_cov = v1.get(art_id, {}).get("verdict", "Missing")
        v2_cov = v2.get(art_id, {}).get("verdict", "Missing")
        delta = COVERAGE_RANK[v1_cov] - COVERAGE_RANK[v2_cov]

        if delta > 0:  # regression — coverage dropped
            c1 = v1.get(art_id, {}).get("final_cited_text") or ""
            c2 = v2.get(art_id, {}).get("final_cited_text") or ""

            if c1 and c2:
                sem_dist = cosine_distance(embedder.embed(c1), embedder.embed(c2))
            elif c1 and not c2:
                # New version lost the citation entirely — maximum distance.
                sem_dist = 1.0
            else:
                # No citation in either version — moderate default.
                sem_dist = 0.5

            risk_w = RISK_WEIGHTS.get(
                v1.get(art_id, {}).get("risk_level", "Medium"), 2
            )
            srs = round(delta * risk_w * (1 + sem_dist), 3)

            regressions.append({
                "article_id": art_id,
                "regulation": v1.get(art_id, {}).get("regulation", ""),
                "from_coverage": v1_cov,
                "to_coverage": v2_cov,
                "risk_level": v1.get(art_id, {}).get("risk_level", "Unknown"),
                "semantic_distance": round(sem_dist, 3),
                "semantic_regression_score": srs,
            })

        elif delta < 0:  # improvement — coverage improved
            improvements.append({
                "article_id": art_id,
                "from": v1_cov,
                "to": v2_cov,
            })

    # Sort regressions so the most severe appear first.
    regressions.sort(key=lambda r: r["semantic_regression_score"], reverse=True)

    return {
        "doc_id": report_v2.get("doc_id", ""),
        "v1_assessed_at": report_v1.get("generated_at", ""),
        "v2_assessed_at": report_v2.get("generated_at", ""),
        "risk_score_v1": report_v1["risk_score"],
        "risk_score_v2": report_v2["risk_score"],
        "risk_score_delta": round(report_v2["risk_score"] - report_v1["risk_score"], 2),
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "critical_regressions": [
            r["article_id"] for r in regressions if r["risk_level"] == "Critical"
        ],
        "max_srs": max(
            (r["semantic_regression_score"] for r in regressions), default=0.0
        ),
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def drift_node(state: dict) -> dict:
    """LangGraph node: runs drift detection if ``previous_report_path`` is set.

    Reads the baseline report from disk, computes drift against the current
    ``violation_report`` in *state*, persists the result to
    ``outputs/drift/{doc_id}_drift_{timestamp}.json``, and returns the drift
    result together with a pipeline-log entry.

    Returns
    -------
    dict
        Keys ``drift_result`` (dict | None) and ``pipeline_log`` (list[dict]).
    """
    prev_path = state.get("previous_report_path")
    if not prev_path:
        return {"drift_result": None, "pipeline_log": []}

    with open(prev_path, encoding="utf-8") as f:
        report_v1 = json.load(f)

    report_v2 = state["violation_report"]
    drift_result = detect_drift(report_v1, report_v2)

    # Persist the drift result to disk.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    drift_path = f"outputs/drift/{state['doc_id']}_drift_{ts}.json"
    os.makedirs(os.path.dirname(drift_path), exist_ok=True)
    with open(drift_path, "w", encoding="utf-8") as f:
        json.dump(drift_result, f, indent=2, ensure_ascii=False)

    # Create pipeline-log entry.
    from backend.logging.pipeline_log import make_log_entry

    log = make_log_entry(
        agent="drift",
        input_data=prev_path,
        raw_prompt=None,
        thinking_trace=None,
        raw_response=None,
        structured_output=drift_result,
    )
    return {"drift_result": drift_result, "pipeline_log": [log]}
