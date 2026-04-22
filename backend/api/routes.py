"""API routes — all endpoints consumed by the frontend.

Endpoints
---------
GET  /api/v1/health
    Returns service status and available regulation namespaces.

POST /api/v1/analyze
    Upload a document (PDF / DOCX / TXT) and run the full compliance pipeline.
    Returns a doc_id the frontend can use to fetch results.

GET  /api/v1/reports/{doc_id}
    Returns the raw ViolationReport JSON for a completed run.

GET  /api/v1/reports/{doc_id}/assessment
    Downloads the Assessment PDF report.

GET  /api/v1/reports/{doc_id}/remediation
    Downloads the Remediation PDF report.

GET  /api/v1/regulations
    Lists all active regulation namespaces and their focus articles.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from backend.agents.classifier import REGULATION_REGISTRY
from backend.evaluation.metrics import (
    compute_hallucination_rate,
    compute_multiclass_metrics,
)
from backend.evaluation.ground_truth import resolve_ground_truth

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_DIR = _PROJECT_ROOT / "outputs" / "reports"
_LOGS_DIR = _PROJECT_ROOT / "outputs" / "logs"


def _latest_pipeline_log_path(doc_id: str) -> Path | None:
    """Return the most recent pipeline JSON log for *doc_id*, if any."""
    if not _LOGS_DIR.exists():
        return None
    matches = list(_LOGS_DIR.glob(f"{doc_id}_*.json"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _evaluation_path(doc_id: str) -> Path:
    """Where the evaluation summary JSON for a run lives."""
    return _REPORTS_DIR / doc_id / "raw" / "evaluation.json"


def _predictions_from_state(state: dict) -> dict:
    """Aggregate predictions across debate records: {article_id: best_verdict}."""
    rank = {"Full": 2, "Partial": 1, "Missing": 0}
    preds: dict[str, str] = {}
    for record in state.get("debate_records", []) or []:
        art_id = record.get("article_id")
        verdict = record.get("verdict", "Missing")
        if not art_id:
            continue
        if art_id not in preds or rank.get(verdict, 0) > rank.get(preds[art_id], 0):
            preds[art_id] = verdict
    return preds


def _kappa_interpretation(kappa: float | None) -> str | None:
    """Landis & Koch (1977) verbal interpretation of Cohen's kappa."""
    if kappa is None:
        return None
    if kappa < 0.0:
        return "poor (worse than chance)"
    if kappa < 0.20:
        return "slight agreement"
    if kappa < 0.40:
        return "fair agreement"
    if kappa < 0.60:
        return "moderate agreement"
    if kappa < 0.80:
        return "substantial agreement"
    return "almost perfect agreement"


def _parse_ground_truth(raw: bytes) -> dict:
    """Accept either a flat ``{article_id: label}`` JSON or an annotated
    ``{article_id: {"label": ...}}`` JSON."""
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Ground truth JSON must be an object")

    # Allow wrappers like {"articles": {...}} or {"annotations": {...}}
    if "articles" in obj and isinstance(obj["articles"], dict):
        obj = obj["articles"]
    elif "annotations" in obj and isinstance(obj["annotations"], dict):
        obj = obj["annotations"]

    flat: dict[str, str] = {}
    for art_id, v in obj.items():
        if isinstance(v, dict):
            label = v.get("label") or v.get("verdict")
        else:
            label = v
        if label:
            flat[str(art_id)] = str(label)
    return flat


def _compute_evaluation(
    state: dict,
    ground_truth: dict | None,
    ground_truth_source: dict | None = None,
) -> dict:
    """Run hallucination + multiclass metrics (if GT) and return a summary.

    ``ground_truth_source`` describes where the GT came from — ``{"source":
    "uploaded"}`` for an explicit upload, or the auto-resolver dict from
    :func:`backend.evaluation.ground_truth.resolve_ground_truth` when the GT
    was pulled from ``test_datasets/*/annotations``.
    """
    debate_records = state.get("debate_records", []) or []
    violation_report = state.get("violation_report") or {}

    # Keep hallucination numbers consistent with the main run summary card by
    # preferring reporter-level (deduplicated/canonical) values.
    if violation_report.get("articles_evaluated") is not None:
        hallucination = {
            "total_evaluations": int(violation_report.get("articles_evaluated", 0) or 0),
            "hallucination_flags": int(violation_report.get("hallucination_flags", 0) or 0),
            "hallucination_rate": float(violation_report.get("hallucination_rate", 0.0) or 0.0),
            "source": "violation_report",
        }
    else:
        hallucination = compute_hallucination_rate(debate_records)
        hallucination["source"] = "debate_records"

    # Restrict scoring to configured focus articles for active regulations.
    # This keeps evaluation fair/aligned with what the pipeline is designed to audit.
    regulation_scope = state.get("regulation_scope", []) or []
    focus_ids: set[str] = set()
    for reg in regulation_scope:
        focus_ids.update(REGULATION_REGISTRY.get(reg, {}).get("focus_articles", []))

    gt_for_scoring = ground_truth or {}
    if focus_ids and gt_for_scoring:
        gt_for_scoring = {k: v for k, v in gt_for_scoring.items() if k in focus_ids}

    # Multiclass metrics if ground truth was provided
    predictions = _predictions_from_state(state)
    if gt_for_scoring:
        classification = compute_multiclass_metrics(predictions, gt_for_scoring)
        classification["kappa_interpretation"] = _kappa_interpretation(
            classification.get("cohens_kappa")
        )
        classification["scoring_scope"] = {
            "mode": "focus_articles_only",
            "regulation_scope": regulation_scope,
            "focus_articles": sorted(focus_ids),
        }
    else:
        classification = {
            "support": 0,
            "macro_f1": None,
            "cohens_kappa": None,
            "accuracy": None,
            "per_class": None,
            "confusion_matrix": None,
            "labels": ["Full", "Partial", "Missing"],
            "kappa_interpretation": None,
            "note": (
                "Ground truth not provided or none overlapped focus_articles "
                "for active regulations"
            ),
            "scoring_scope": {
                "mode": "focus_articles_only",
                "regulation_scope": regulation_scope,
                "focus_articles": sorted(focus_ids),
            },
        }

    return {
        "hallucination": hallucination,
        "classification": classification,
        "predictions": predictions,
        "n_predictions": len(predictions),
        "ground_truth_provided": bool(ground_truth),
        "ground_truth_source": ground_truth_source or {"source": "none"},
        "ground_truth": ground_truth or {},
        "ground_truth_scored": gt_for_scoring,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    """Service health check — returns active regulation namespaces."""
    active = [
        k for k, v in REGULATION_REGISTRY.items()
        if v.get("status") == "active"
    ]
    return {"status": "ok", "active_regulations": active}


# ---------------------------------------------------------------------------
# Regulations
# ---------------------------------------------------------------------------

@router.get("/regulations")
def list_regulations():
    """Return all active regulation namespaces with their focus articles."""
    return {
        k: {
            "namespace": v["namespace"],
            "focus_articles": v["focus_articles"],
        }
        for k, v in REGULATION_REGISTRY.items()
        if v.get("status") == "active"
    }


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    ground_truth: UploadFile | None = File(None),
):
    """Upload a document and run the full compliance pipeline.

    Accepts PDF, DOCX, or TXT.  Optionally accepts a JSON *ground_truth* file
    whose body is either ``{article_id: label}`` or
    ``{article_id: {"label": ...}}``; when supplied, per-class precision/recall/F1,
    macro-F1, confusion matrix and multi-class Cohen's kappa are computed.
    Hallucination rate is always computed.
    """
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: .pdf, .docx, .txt",
        )

    # Optional ground truth — read before running the pipeline
    gt_dict: dict | None = None
    gt_source: dict | None = None
    if ground_truth is not None and ground_truth.filename:
        try:
            raw = await ground_truth.read()
            gt_dict = _parse_ground_truth(raw) if raw else None
            if gt_dict:
                gt_source = {
                    "source": "uploaded",
                    "filename": ground_truth.filename,
                    "n_articles": len(gt_dict),
                }
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ground_truth JSON: {e}",
            ) from e

    # Auto-resolve ground truth from test_datasets annotation PDFs when the
    # user did not upload one explicitly.
    if not gt_dict and file.filename:
        try:
            auto_gt, auto_info = resolve_ground_truth(file.filename)
            if auto_gt:
                gt_dict = auto_gt
                gt_source = auto_info
        except Exception:  # noqa: BLE001 — GT resolution must never break analysis
            gt_source = gt_source or {"source": "none"}

    # Write upload to a temp file so the pipeline can read it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from backend.graph import run_pipeline
        state = run_pipeline(tmp_path)
    finally:
        os.unlink(tmp_path)

    doc_id = state["doc_id"]
    vr = state.get("violation_report", {})

    # Evaluation metrics (hallucination + optional multiclass)
    evaluation = _compute_evaluation(state, gt_dict, gt_source)

    # Persist evaluation summary alongside the violation report
    try:
        eval_path = _evaluation_path(doc_id)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as fh:
            json.dump(evaluation, fh, indent=2)
    except Exception:  # noqa: BLE001 — non-fatal, keep the response
        pass

    log_path = _latest_pipeline_log_path(doc_id)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "doc_type": state.get("doc_type"),
        "regulations": state.get("regulation_scope", []),
        "risk_score": state.get("risk_score"),
        "risk_level": state.get("risk_level"),
        "articles_evaluated": vr.get("articles_evaluated"),
        "hallucination_rate": vr.get("hallucination_rate"),
        "evaluation": evaluation,
        "evaluation_url": f"/api/v1/reports/{doc_id}/evaluation",
        "violation_report_url": f"/api/v1/reports/{doc_id}",
        # Include article-level rows directly so UI can render findings even if
        # the follow-up report fetch fails transiently.
        "violations": vr.get("violations", []),
        "assessment_report_url": f"/api/v1/reports/{doc_id}/assessment",
        "remediation_report_url": f"/api/v1/reports/{doc_id}/remediation",
        "pipeline_log_url": f"/api/v1/reports/{doc_id}/pipeline-log"
        if log_path and log_path.exists()
        else None,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@router.get("/reports/{doc_id}")
def get_report(doc_id: str):
    """Return the raw ViolationReport JSON for a completed analysis run."""
    report_path = _REPORTS_DIR / doc_id / "raw" / "violation_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"No report found for doc_id '{doc_id}'")
    with open(report_path, encoding="utf-8") as fh:
        return JSONResponse(content=json.load(fh))


@router.get("/reports/{doc_id}/evaluation")
def get_evaluation(doc_id: str):
    """Return the evaluation summary (hallucination + classification metrics)."""
    eval_path = _evaluation_path(doc_id)
    if not eval_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation summary found for doc_id '{doc_id}'",
        )
    with open(eval_path, encoding="utf-8") as fh:
        return JSONResponse(content=json.load(fh))


@router.get("/reports/{doc_id}/assessment")
def get_assessment_pdf(doc_id: str):
    """Download the Assessment PDF report."""
    pdf_path = _REPORTS_DIR / doc_id / "POA&M" / "assessment_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Assessment report not found")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"assessment_report_{doc_id}.pdf",
    )


@router.get("/reports/{doc_id}/pipeline-log")
def get_pipeline_log_json(doc_id: str):
    """Download the full pipeline log JSON (thinking traces, prompts) for a run."""
    log_path = _latest_pipeline_log_path(doc_id)
    if not log_path or not log_path.exists():
        raise HTTPException(status_code=404, detail="Pipeline log not found for this analysis")
    return FileResponse(
        path=str(log_path),
        media_type="application/json",
        filename=log_path.name,
    )


@router.get("/reports/{doc_id}/remediation")
def get_remediation_pdf(doc_id: str):
    """Download the Remediation PDF report."""
    pdf_path = _REPORTS_DIR / doc_id / "POA&M" / "remediation_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Remediation report not found")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"remediation_report_{doc_id}.pdf",
    )


@router.get("/reports")
def list_reports():
    """List all completed analysis runs (doc_ids with their risk summary)."""
    reports = []
    if _REPORTS_DIR.exists():
        for doc_dir in sorted(_REPORTS_DIR.iterdir()):
            raw_path = doc_dir / "raw" / "violation_report.json"
            if raw_path.exists():
                with open(raw_path, encoding="utf-8") as fh:
                    vr = json.load(fh)
                reports.append({
                    "doc_id": doc_dir.name,
                    "doc_type": vr.get("doc_type"),
                    "regulations": vr.get("regulations", []),
                    "risk_score": vr.get("risk_score"),
                    "risk_level": vr.get("risk_level"),
                    "generated_at": vr.get("generated_at"),
                })
    return {"reports": reports}
