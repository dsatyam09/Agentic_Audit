"""ReporterAgent — risk scoring, violation report assembly, and POA&M generation.

Reads debate_records from state, deduplicates per article, computes risk scores,
generates remediation text via Qwen3-8B, and renders POA&M reports (Assessment +
Remediation) using Jinja2 templates.

File: backend/agents/reporter.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from backend.agents.state import DebateRecord, POAMReport
from backend.logging.pipeline_log import make_log_entry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_WEIGHTS = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
COVERAGE_PENALTY = {"Missing": 1.0, "Partial": 0.5, "Full": 0.0}
RISK_THRESHOLDS = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Critical": 4.0}

VERDICT_RANK = {"Full": 0, "Partial": 1, "Missing": 2}

REMEDIATION_PROMPT = """\
You are a GDPR/compliance consultant. A compliance audit has identified these violations.
For each violation, write the exact language that should be added to the policy document.

VIOLATIONS:
{violations_json}

For each violation, provide:
1. The exact clause text to add (2-3 sentences, ready to paste into a policy document)
2. Which section of the document it should appear in

Respond as JSON array:
[{{"article_id": "art_17", "section": "User Rights", "remediation_text": "Users may request the deletion of their personal data by contacting our Data Protection Officer at [dpo@company.com]. We will process all valid erasure requests within 30 days of receipt, subject to legal retention obligations under applicable law. To submit an erasure request, please provide your account details and specify the data you wish to have deleted."}}]
"""


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def compute_risk_score(canonical_records: list[dict]) -> tuple[float, str]:
    """Compute aggregate risk score from deduplicated per-article debate records.

    Parameters
    ----------
    canonical_records : list[dict]
        One record per article (best verdict already selected via deduplication).

    Returns
    -------
    tuple[float, str]
        (score, level) where score is in [0.0, 4.0] and level is one of
        "Low", "Medium", "High", "Critical".
    """
    if not canonical_records:
        return 0.0, "Low"

    total = sum(
        RISK_WEIGHTS.get(r["risk_level"], 1) * COVERAGE_PENALTY.get(r["verdict"], 1.0)
        for r in canonical_records
    )
    max_possible = sum(RISK_WEIGHTS.get(r["risk_level"], 1) for r in canonical_records)
    score = round(total / max_possible * 4, 2) if max_possible else 0.0

    level = next(
        k
        for k, v in sorted(RISK_THRESHOLDS.items(), key=lambda x: x[1])
        if score <= v
    )
    return score, level


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_records(debate_records: list[dict]) -> list[dict]:
    """Deduplicate debate records so there is one canonical record per article_id.

    When the same article_id appears from multiple chunks, we keep the record
    with the **best** verdict: Full > Partial > Missing.  Ties on verdict are
    broken by lowest chunk_index (earliest occurrence).
    """
    best: dict[str, dict] = {}
    for rec in debate_records:
        aid = rec["article_id"]
        if aid not in best:
            best[aid] = rec
        else:
            current_rank = VERDICT_RANK.get(best[aid]["verdict"], 2)
            new_rank = VERDICT_RANK.get(rec["verdict"], 2)
            if new_rank < current_rank:
                best[aid] = rec
            elif new_rank == current_rank and rec.get("chunk_index", 0) < best[aid].get("chunk_index", 0):
                best[aid] = rec
    return list(best.values())


# ---------------------------------------------------------------------------
# Remediation generation via Qwen3-8B
# ---------------------------------------------------------------------------

def _generate_remediations(
    violations: list[dict],
    qwen_runner=None,
) -> tuple[dict[str, str], str, str]:
    """Call Qwen3-8B once to generate remediation text for all violations.

    Parameters
    ----------
    violations : list[dict]
        Violation dicts (only Missing/Partial entries).
    qwen_runner : QwenRunner | None
        Qwen runner instance.  Imported lazily to avoid loading the model at
        import time.

    Returns
    -------
    tuple[dict[str, str], str, str]
        (remediation_map, thinking_trace, full_output) where remediation_map
        is a mapping of article_id -> remediation text.
    """
    if not violations:
        return {}, "", ""

    # Lazy import so the heavy model is only loaded when actually needed
    if qwen_runner is None:
        from backend.debate.qwen_runner import qwen as _qwen
        qwen_runner = _qwen

    violations_json = json.dumps(
        [
            {
                "article_id": v["article_id"],
                "article_title": v["article_title"],
                "regulation": v["regulation"],
                "verdict": v["verdict"],
                "risk_level": v["risk_level"],
                "reasoning": v["reasoning"],
                "challenger_gap": v.get("challenger_gap", ""),
            }
            for v in violations
        ],
        indent=2,
    )

    prompt = REMEDIATION_PROMPT.format(violations_json=violations_json)
    result = qwen_runner.generate(prompt, thinking=True, max_new_tokens=2048)

    # Parse the JSON response
    remediation_map: dict[str, str] = {}
    try:
        parsed = json.loads(result["response"])
        if isinstance(parsed, list):
            for item in parsed:
                aid = item.get("article_id", "")
                text = item.get("remediation_text", "")
                remediation_map[aid] = text
    except (json.JSONDecodeError, TypeError):
        # Fallback: try to extract JSON array from response
        response_text = result["response"]
        start = response_text.find("[")
        end = response_text.rfind("]")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(response_text[start : end + 1])
                for item in parsed:
                    aid = item.get("article_id", "")
                    text = item.get("remediation_text", "")
                    remediation_map[aid] = text
            except (json.JSONDecodeError, TypeError):
                pass

    # Provide fallback remediation for any violation that was not covered
    for v in violations:
        if v["article_id"] not in remediation_map:
            remediation_map[v["article_id"]] = (
                f"Address the gap identified for {v['article_id']} ({v['article_title']}). "
                f"Ensure the document explicitly covers the requirements of this article "
                f"as specified by {v['regulation'].upper()} regulations."
            )

    return remediation_map, result.get("thinking_trace", ""), result.get("full_output", "")


# ---------------------------------------------------------------------------
# POA&M report generation
# ---------------------------------------------------------------------------

def generate_poam(
    violation_report: dict,
    debate_records: list[dict],
    state: dict,
    doc_id: str,
) -> POAMReport:
    """Render both Jinja2 templates and write PDF files to outputs/reports/{doc_id}/POA&M/.

    Returns
    -------
    POAMReport
        Paths to the two generated PDF report files.
    """
    from backend.reports.assessment import render_assessment_report
    from backend.reports.remediation import render_remediation_report
    from backend.reports.pdf_renderer import markdown_to_pdf

    assessment_dir = _PROJECT_ROOT / "outputs" / "reports" / doc_id / "POA&M"
    os.makedirs(assessment_dir, exist_ok=True)

    assessment_path = str(assessment_dir / "assessment_report.pdf")
    remediation_path = str(assessment_dir / "remediation_report.pdf")

    # Render assessment report markdown then convert to PDF
    assessment_md = render_assessment_report(violation_report, debate_records, state)
    markdown_to_pdf(assessment_md, assessment_path)

    # Render remediation report markdown then convert to PDF
    remediation_md = render_remediation_report(violation_report, state)
    markdown_to_pdf(remediation_md, remediation_path)

    return POAMReport(
        assessment_report_path=assessment_path,
        remediation_report_path=remediation_path,
    )


# ---------------------------------------------------------------------------
# Build violation report
# ---------------------------------------------------------------------------

def _build_violation_report(
    canonical_records: list[dict],
    risk_score: float,
    risk_level: str,
    doc_id: str,
    doc_type: str,
    regulation_scope: list[str],
    remediation_map: dict[str, str],
    generated_at: str,
) -> dict:
    """Construct the ViolationReport dict matching the spec Section 8 schema."""
    violations = []
    hallucination_count = 0

    for rec in canonical_records:
        is_hallucinated = rec.get("hallucination_flag", False)
        if is_hallucinated:
            hallucination_count += 1

        violation_entry = {
            "article_id": rec["article_id"],
            "article_title": rec["article_title"],
            "regulation": rec["regulation"],
            "verdict": rec["verdict"],
            "risk_level": rec["risk_level"],
            "reasoning": rec["reasoning"],
            "final_cited_text": rec.get("final_cited_text"),
            "debate_summary": rec.get("debate_summary", ""),
            "remediation": remediation_map.get(rec["article_id"], ""),
            "hallucination_flag": is_hallucinated,
        }
        violations.append(violation_entry)

    n_evaluated = len(canonical_records)
    hallucination_rate = round(hallucination_count / n_evaluated, 4) if n_evaluated else 0.0

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "regulations": regulation_scope,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "articles_evaluated": n_evaluated,
        "violations": violations,
        "hallucination_flags": hallucination_count,
        "hallucination_rate": hallucination_rate,
        "generated_at": generated_at,
        "model": "Qwen/Qwen3-8B",
        "regulation_versions": {},
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def reporter_node(state: dict) -> dict:
    """LangGraph node: computes risk, generates remediation, builds reports.

    Reads
    -----
    state["debate_records"], state["doc_id"], state["doc_type"],
    state["regulation_scope"]

    Writes
    ------
    state["risk_score"], state["risk_level"], state["violation_report"],
    state["poam"], state["pipeline_log"]
    """
    debate_records: list[dict] = state.get("debate_records", [])
    doc_id: str = state.get("doc_id", "unknown")
    doc_type: str = state.get("doc_type", "unknown")
    regulation_scope: list[str] = state.get("regulation_scope", [])

    generated_at = datetime.now(timezone.utc).isoformat()

    # ── Step 1: Deduplicate debate records ─────────────────────────────────
    canonical_records = deduplicate_records(debate_records)

    # ── Step 2: Compute risk score ─────────────────────────────────────────
    risk_score, risk_level = compute_risk_score(canonical_records)

    # ── Step 3: Generate remediation for violations (Missing / Partial) ────
    violations_needing_remediation = [
        rec for rec in canonical_records if rec["verdict"] in ("Missing", "Partial")
    ]

    remediation_map: dict[str, str] = {}
    remediation_thinking = ""
    remediation_raw = ""

    if violations_needing_remediation:
        try:
            remediation_map, remediation_thinking, remediation_raw = _generate_remediations(
                violations_needing_remediation
            )
        except Exception as e:
            # If Qwen runner fails (e.g., model not loaded), provide fallback
            for v in violations_needing_remediation:
                remediation_map[v["article_id"]] = (
                    f"Review and update the document to address {v['article_id']} "
                    f"({v['article_title']}). The current coverage is '{v['verdict']}'. "
                    f"Ensure all key requirements of this article are explicitly addressed."
                )
            remediation_thinking = ""
            remediation_raw = f"Error generating remediations: {e}"

    # ── Step 4: Build violation report ─────────────────────────────────────
    violation_report = _build_violation_report(
        canonical_records=canonical_records,
        risk_score=risk_score,
        risk_level=risk_level,
        doc_id=doc_id,
        doc_type=doc_type,
        regulation_scope=regulation_scope,
        remediation_map=remediation_map,
        generated_at=generated_at,
    )

    # ── Step 5: Generate POA&M reports ─────────────────────────────────────
    poam = generate_poam(
        violation_report=violation_report,
        debate_records=debate_records,
        state=state,
        doc_id=doc_id,
    )

    # ── Step 6: Write raw violation report JSON ────────────────────────────
    raw_dir = _PROJECT_ROOT / "outputs" / "reports" / doc_id / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = raw_dir / "violation_report.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(violation_report, fh, indent=2, ensure_ascii=False)

    # ── Step 7: Log entry ──────────────────────────────────────────────────
    log_entry = make_log_entry(
        agent="reporter",
        input_data={
            "doc_id": doc_id,
            "debate_records_count": len(debate_records),
            "canonical_records_count": len(canonical_records),
        },
        raw_prompt=REMEDIATION_PROMPT.format(violations_json="[...]") if violations_needing_remediation else None,
        thinking_trace=remediation_thinking if remediation_thinking else None,
        raw_response=remediation_raw if remediation_raw else None,
        structured_output={
            "risk_score": risk_score,
            "risk_level": risk_level,
            "articles_evaluated": len(canonical_records),
            "violations_count": len([v for v in violation_report["violations"] if v["verdict"] != "Full"]),
            "hallucination_flags": violation_report["hallucination_flags"],
            "assessment_report_path": poam["assessment_report_path"],
            "remediation_report_path": poam["remediation_report_path"],
        },
    )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "violation_report": violation_report,
        "poam": poam,
        "pipeline_log": [log_entry],
    }
