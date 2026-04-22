"""Remediation Report renderer — generates the POA&M Remediation Report.

Renders the Jinja2 template at ``templates/remediation.md.jinja`` with data
from the violation report and pipeline state.

File: backend/reports/remediation.py
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _build_template_env() -> Environment:
    """Create a Jinja2 environment pointed at the templates directory."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _estimate_effort(n_critical: int, n_high: int, n_medium: int) -> str:
    """Derive a human-readable effort estimate from severity counts."""
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


def render_remediation_report(
    violation_report: dict,
    state: dict,
) -> str:
    """Render the Remediation Report markdown from the Jinja2 template.

    Parameters
    ----------
    violation_report : dict
        The ViolationReport dict (Section 8 schema).
    state : dict
        Full pipeline state (used to pull drift data, challenger gaps, etc.).

    Returns
    -------
    str
        Rendered markdown string.
    """
    env = _build_template_env()
    template = env.get_template("remediation.md.jinja")

    violations = violation_report.get("violations", [])
    regulation_scope = violation_report.get("regulations", [])
    regulation_scope_str = ", ".join(r.upper() for r in regulation_scope)

    # Filter to only Missing/Partial — these need remediation
    actionable = [v for v in violations if v.get("verdict") in ("Missing", "Partial")]

    # Sort by risk level descending (Critical first)
    risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    actionable_sorted = sorted(
        actionable,
        key=lambda v: (risk_order.get(v.get("risk_level", "Low"), 4), v.get("article_id", "")),
    )

    # Severity counts
    n_critical = sum(1 for v in actionable_sorted if v.get("risk_level") == "Critical")
    n_high = sum(1 for v in actionable_sorted if v.get("risk_level") == "High")
    n_medium = sum(1 for v in actionable_sorted if v.get("risk_level") == "Medium")
    n_low = sum(1 for v in actionable_sorted if v.get("risk_level") == "Low")

    effort_estimate = _estimate_effort(n_critical, n_high, n_medium)

    # Build a lookup from article_id -> challenger_gap from debate records
    debate_records = state.get("debate_records", [])
    challenger_gaps: dict[str, str] = {}
    for rec in debate_records:
        aid = rec.get("article_id", "")
        gap = rec.get("challenger_gap", "")
        # Keep the most informative gap (longest text) per article
        if aid and gap and len(gap) > len(challenger_gaps.get(aid, "")):
            challenger_gaps[aid] = gap

    # Enrich each actionable violation with challenger_gap
    for v in actionable_sorted:
        v["challenger_gap"] = challenger_gaps.get(v.get("article_id", ""), "Gap details not available.")

    # Drift result from state
    drift_result = state.get("drift_result")

    rendered = template.render(
        doc_id=violation_report.get("doc_id", ""),
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
