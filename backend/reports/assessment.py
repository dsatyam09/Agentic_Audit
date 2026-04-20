"""Assessment Report renderer — generates the POA&M Assessment Report.

Renders the Jinja2 template at ``templates/assessment.md.jinja`` with data
from the violation report, debate records, and pipeline state.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _build_template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _build_executive_summary(violation_report: dict) -> str:
    """Generate a concise executive summary paragraph."""
    doc_name = (
        violation_report.get("doc_filename")
        or violation_report.get("doc_id", "Unknown")
    )
    doc_type = violation_report.get("doc_type", "document")
    risk_score = violation_report.get("risk_score", 0.0)
    risk_level = violation_report.get("risk_level", "Low")
    n_evaluated = violation_report.get("articles_evaluated", 0)
    regulations = ", ".join(
        r.upper() for r in violation_report.get("regulations", [])
    )

    violations = violation_report.get("violations", [])
    n_full = sum(1 for v in violations if v.get("verdict") == "Full")
    n_partial = sum(1 for v in violations if v.get("verdict") == "Partial")
    n_missing = sum(1 for v in violations if v.get("verdict") == "Missing")

    critical_gaps = [
        v for v in violations
        if v.get("verdict") in ("Missing", "Partial") and v.get("risk_level") == "Critical"
    ]
    high_gaps = [
        v for v in violations
        if v.get("verdict") in ("Missing", "Partial") and v.get("risk_level") == "High"
    ]

    lines = [
        f"This assessment evaluated document **{doc_name}** (type: {doc_type}) "
        f"against {n_evaluated} articles from {regulations}. "
        f"The overall risk score is **{risk_score} / 4.0** ({risk_level} risk).",
        "",
        f"Of the {n_evaluated} articles evaluated, {n_full} received Full coverage, "
        f"{n_partial} received Partial coverage, and {n_missing} were found Missing.",
    ]

    if critical_gaps:
        article_list = ", ".join(v["article_id"] for v in critical_gaps)
        lines.append(
            f"**Critical gaps** were identified in: {article_list}. "
            f"These require immediate remediation to avoid regulatory exposure."
        )

    if high_gaps:
        article_list = ", ".join(v["article_id"] for v in high_gaps)
        lines.append(
            f"**High-priority gaps** were identified in: {article_list}. "
            f"These should be addressed in the next document revision cycle."
        )

    if not critical_gaps and not high_gaps:
        lines.append(
            "No critical or high-priority gaps were identified. "
            "The document demonstrates strong regulatory coverage."
        )

    return "\n".join(lines)


def render_assessment_report(
    violation_report: dict,
    debate_records: list[dict],
    state: dict,
) -> str:
    """Render the Assessment Report markdown from the Jinja2 template."""
    env = _build_template_env()
    template = env.get_template("assessment.md.jinja")

    summary = _build_executive_summary(violation_report)

    reg_versions = violation_report.get("regulation_versions", {})
    regulation_scope = violation_report.get("regulations", [])
    regulation_scope_str = ", ".join(r.upper() for r in regulation_scope)

    risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    sorted_violations = sorted(
        violation_report.get("violations", []),
        key=lambda v: (risk_order.get(v.get("risk_level", "Low"), 4), v.get("article_id", "")),
    )

    drift_result = state.get("drift_result")

    rendered = template.render(
        doc_id=violation_report.get("doc_id", ""),
        doc_filename=violation_report.get("doc_filename", ""),
        doc_sha256=violation_report.get("doc_sha256", ""),
        doc_type=violation_report.get("doc_type", ""),
        regulation_scope=regulation_scope_str,
        generated_at=violation_report.get("generated_at", ""),
        risk_score=violation_report.get("risk_score", 0.0),
        risk_level=violation_report.get("risk_level", "Low"),
        model=violation_report.get("model", "Qwen/Qwen3-8B"),
        thinking_enabled=violation_report.get("thinking_enabled", True),
        summary=summary,
        regulation_versions=reg_versions,
        violations=sorted_violations,
        articles_evaluated=violation_report.get("articles_evaluated", 0),
        hallucination_flags=violation_report.get("hallucination_flags", 0),
        hallucination_rate=violation_report.get("hallucination_rate", 0.0),
        debate_records=debate_records,
        drift_result=drift_result,
    )

    return rendered
