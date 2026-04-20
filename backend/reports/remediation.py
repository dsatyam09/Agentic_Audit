"""Remediation Report renderer — generates the POA&M Remediation Report."""

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


def _estimate_effort(n_critical: int, n_high: int, n_medium: int) -> str:
    total_weight = n_critical * 4 + n_high * 3 + n_medium * 2
    if total_weight == 0:
        return "Minimal — no remediation required"
    elif total_weight <= 4:
        return "Low — estimated 1-2 hours of policy revision"
    elif total_weight <= 10:
        return "Moderate — estimated 3-5 hours of policy revision and legal review"
    elif total_weight <= 20:
        return "High — estimated 1-2 days of policy revision, legal review, and stakeholder sign-off"
    else:
        return "Critical — estimated 3-5 days of comprehensive policy overhaul, legal counsel engagement, and executive review"


def render_remediation_report(violation_report: dict, state: dict) -> str:
    """Render the Remediation Report markdown from the Jinja2 template."""
    env = _build_template_env()
    template = env.get_template("remediation.md.jinja")

    violations = violation_report.get("violations", [])
    regulation_scope = violation_report.get("regulations", [])
    regulation_scope_str = ", ".join(r.upper() for r in regulation_scope)

    actionable = [v for v in violations if v.get("verdict") in ("Missing", "Partial")]

    risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    actionable_sorted = sorted(
        actionable,
        key=lambda v: (risk_order.get(v.get("risk_level", "Low"), 4), v.get("article_id", "")),
    )

    n_critical = sum(1 for v in actionable_sorted if v.get("risk_level") == "Critical")
    n_high = sum(1 for v in actionable_sorted if v.get("risk_level") == "High")
    n_medium = sum(1 for v in actionable_sorted if v.get("risk_level") == "Medium")
    n_low = sum(1 for v in actionable_sorted if v.get("risk_level") == "Low")

    effort_estimate = _estimate_effort(n_critical, n_high, n_medium)

    # Ensure each actionable violation has a challenger_gap field (the
    # violation_report already includes it; fall back to debate_records
    # for legacy reports).
    debate_records = state.get("debate_records", [])
    challenger_gaps: dict[str, str] = {}
    for rec in debate_records:
        aid = rec.get("article_id", "")
        gap = rec.get("challenger_gap", "")
        if aid and gap and len(gap) > len(challenger_gaps.get(aid, "")):
            challenger_gaps[aid] = gap
    for v in actionable_sorted:
        if not v.get("challenger_gap"):
            v["challenger_gap"] = challenger_gaps.get(v.get("article_id", ""), "")

    drift_result = state.get("drift_result")

    rendered = template.render(
        doc_id=violation_report.get("doc_id", ""),
        doc_filename=violation_report.get("doc_filename", ""),
        regulation_scope=regulation_scope_str,
        risk_level=violation_report.get("risk_level", "Low"),
        generated_at=violation_report.get("generated_at", ""),
        n_violations=len(actionable_sorted),
        n_critical=n_critical,
        n_high=n_high,
        n_medium=n_medium,
        n_low=n_low,
        effort_estimate=effort_estimate,
        actions=actionable_sorted,
        drift_result=drift_result,
    )

    return rendered
