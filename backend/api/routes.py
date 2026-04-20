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

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_DIR = _PROJECT_ROOT / "outputs" / "reports"


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
async def analyze(file: UploadFile = File(...), thinking: bool = True):
    """Upload a document and run the full compliance pipeline.

    Accepts PDF, DOCX, or TXT.  Returns a summary and the doc_id so the
    caller can fetch the full report via GET /reports/{doc_id}.

    Set ``thinking=false`` as a query param to run the C4-nothink ablation.
    """
    original_filename = file.filename or "upload"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: .pdf, .docx, .txt",
        )

    # Preserve the original filename so it appears in the generated reports
    # rather than a random tmp name.
    tmp_dir = tempfile.mkdtemp(prefix="agentic_audit_")
    tmp_path = os.path.join(tmp_dir, original_filename)
    with open(tmp_path, "wb") as fh:
        fh.write(await file.read())

    try:
        from backend.graph import run_pipeline
        state = run_pipeline(tmp_path, thinking=thinking)
    finally:
        try:
            os.unlink(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    doc_id = state["doc_id"]
    vr = state.get("violation_report", {})

    return {
        "doc_id": doc_id,
        "doc_filename": state.get("doc_filename", original_filename),
        "doc_type": state.get("doc_type"),
        "regulations": state.get("regulation_scope", []),
        "risk_score": state.get("risk_score"),
        "risk_level": state.get("risk_level"),
        "articles_evaluated": vr.get("articles_evaluated"),
        "hallucination_rate": vr.get("hallucination_rate"),
        "thinking_enabled": vr.get("thinking_enabled", thinking),
        "assessment_report_url": f"/api/v1/reports/{doc_id}/assessment",
        "remediation_report_url": f"/api/v1/reports/{doc_id}/remediation",
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


def _download_filename(doc_id: str, kind: str) -> str:
    """Prefer the original document's stem + kind + doc_id for download names."""
    raw_path = _REPORTS_DIR / doc_id / "raw" / "violation_report.json"
    stem = doc_id
    if raw_path.exists():
        try:
            with open(raw_path, encoding="utf-8") as fh:
                vr = json.load(fh)
            original = vr.get("doc_filename", "")
            if original:
                stem = Path(original).stem or doc_id
        except (json.JSONDecodeError, OSError):
            pass
    return f"{stem}_{kind}_{doc_id}.pdf"


@router.get("/reports/{doc_id}/assessment")
def get_assessment_pdf(doc_id: str):
    """Download the Assessment PDF report."""
    pdf_path = _REPORTS_DIR / doc_id / "POA&M" / "assessment_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Assessment report not found")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=_download_filename(doc_id, "assessment"),
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
        filename=_download_filename(doc_id, "remediation"),
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
