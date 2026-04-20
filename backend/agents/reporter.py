"""ReporterAgent — risk scoring, violation report assembly, and POA&M generation.

The reporter runs in two phases (two LangGraph nodes):

1. ``reporter_compute_node``: deduplicates debate records, computes risk,
   generates remediation text, builds the ViolationReport dict, and
   persists it to ``outputs/reports/{doc_id}/raw/violation_report.json``.

2. ``reporter_render_node``: renders the Assessment + Remediation Jinja
   templates to PDFs. Runs *after* drift detection so PDFs always include
   the latest drift data when a previous report was supplied.
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
_COMPLIANCE_DATA_DIR = _PROJECT_ROOT / "data" / "compliance"

RISK_WEIGHTS = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
COVERAGE_PENALTY = {"Missing": 1.0, "Partial": 0.5, "Full": 0.0}
RISK_THRESHOLDS = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Critical": 4.0}

VERDICT_RANK = {"Full": 0, "Partial": 1, "Missing": 2}

REMEDIATION_PROMPT = """\
You are a compliance consultant. A compliance audit has identified the following
violations. For each violation, write the exact language that should be added to
the policy document.

VIOLATIONS:
{violations_json}

For each violation, provide:
1. The exact clause text to add (2-3 sentences, ready to paste into a policy document).
2. The section of the document it should appear in.

Respond with ONLY a JSON array — no prose, no markdown fences:
[{{"article_id": "<id>", "section": "<section name>", "remediation_text": "<ready-to-paste language>"}}]
"""


# ---------------------------------------------------------------------------
# Cached article metadata (for key_requirements + clause_text in templates)
# ---------------------------------------------------------------------------

_ARTICLE_META_CACHE: dict[str, dict[str, dict]] = {}


def _load_article_meta(regulation: str) -> dict[str, dict]:
    """Return {article_id: full_article_dict} for a regulation, cached."""
    if regulation in _ARTICLE_META_CACHE:
        return _ARTICLE_META_CACHE[regulation]

    path = _COMPLIANCE_DATA_DIR / regulation / f"{regulation}_articles.json"
    mapping: dict[str, dict] = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                articles = json.load(fh)
            mapping = {a["article_id"]: a for a in articles if "article_id" in a}
        except (json.JSONDecodeError, OSError):
            mapping = {}

    _ARTICLE_META_CACHE[regulation] = mapping
    return mapping


def _article_meta(regulation: str, article_id: str) -> dict:
    """Return a single article's metadata, or {} if not found."""
    return _load_article_meta(regulation).get(article_id, {})


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def compute_risk_score(canonical_records: list[dict]) -> tuple[float, str]:
    """Compute aggregate risk score from deduplicated per-article debate records."""
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

    Preference order:
    1. Best verdict (Full > Partial > Missing).
    2. Prefer non-hallucinated records over hallucinated ones at the same verdict.
    3. Earliest chunk_index as final tiebreaker.
    """
    best: dict[str, dict] = {}
    for rec in debate_records:
        aid = rec["article_id"]
        if aid not in best:
            best[aid] = rec
            continue

        current = best[aid]
        current_rank = VERDICT_RANK.get(current["verdict"], 2)
        new_rank = VERDICT_RANK.get(rec["verdict"], 2)

        if new_rank < current_rank:
            best[aid] = rec
            continue
        if new_rank > current_rank:
            continue

        # Same verdict — prefer non-hallucinated.
        current_hallucinated = current.get("hallucination_flag", False)
        new_hallucinated = rec.get("hallucination_flag", False)
        if current_hallucinated and not new_hallucinated:
            best[aid] = rec
            continue
        if new_hallucinated and not current_hallucinated:
            continue

        # Final tiebreaker: earliest chunk_index.
        if rec.get("chunk_index", 0) < current.get("chunk_index", 0):
            best[aid] = rec

    return list(best.values())


# ---------------------------------------------------------------------------
# Remediation generation via Qwen3-8B
# ---------------------------------------------------------------------------

def _generate_remediations(
    violations: list[dict],
    qwen_runner=None,
    thinking: bool = True,
) -> tuple[dict[str, str], str, str]:
    """Call Qwen3-8B once to generate remediation text for all violations."""
    if not violations:
        return {}, "", ""

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
    result = qwen_runner.generate(prompt, thinking=thinking, max_new_tokens=2048)

    remediation_map: dict[str, str] = {}
    response_text = result["response"]

    def _ingest(items: list) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            aid = item.get("article_id", "")
            text = item.get("remediation_text", "")
            if aid and text:
                remediation_map[aid] = text

    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, list):
            _ingest(parsed)
    except (json.JSONDecodeError, TypeError):
        start = response_text.find("[")
        end = response_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(response_text[start : end + 1])
                if isinstance(parsed, list):
                    _ingest(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

    # Fill in any gaps with a deterministic fallback so every violation has
    # something actionable in the remediation report.
    for v in violations:
        if v["article_id"] not in remediation_map or not remediation_map[v["article_id"]]:
            remediation_map[v["article_id"]] = (
                f"Address the gap identified for {v['article_id']} ({v['article_title']}). "
                f"Add explicit language to the document covering the requirements of this "
                f"article under {v['regulation'].upper()}."
            )

    return remediation_map, result.get("thinking_trace", ""), result.get("full_output", "")


# ---------------------------------------------------------------------------
# Build violation report
# ---------------------------------------------------------------------------

def _build_violation_report(
    *,
    canonical_records: list[dict],
    risk_score: float,
    risk_level: str,
    doc_id: str,
    doc_filename: str,
    doc_path: str,
    doc_sha256: str,
    doc_type: str,
    regulation_scope: list[str],
    remediation_map: dict[str, str],
    generated_at: str,
    thinking_enabled: bool,
) -> dict:
    """Construct the ViolationReport dict matching the spec Section 8 schema."""
    violations = []
    hallucination_count = 0

    for rec in canonical_records:
        is_hallucinated = rec.get("hallucination_flag", False)
        if is_hallucinated:
            hallucination_count += 1

        meta = _article_meta(rec.get("regulation", ""), rec["article_id"])
        violation_entry = {
            "article_id": rec["article_id"],
            "article_title": rec["article_title"],
            "regulation": rec["regulation"],
            "clause_text": meta.get("content", ""),
            "key_requirements": meta.get("key_requirements", []),
            "verdict": rec["verdict"],
            "risk_level": rec["risk_level"],
            "reasoning": rec["reasoning"],
            "final_cited_text": rec.get("final_cited_text"),
            "debate_summary": rec.get("debate_summary", ""),
            "challenger_gap": rec.get("challenger_gap", ""),
            "remediation": remediation_map.get(rec["article_id"], ""),
            "hallucination_flag": is_hallucinated,
            "chunk_index": rec.get("chunk_index"),
        }
        violations.append(violation_entry)

    n_evaluated = len(canonical_records)
    hallucination_rate = round(hallucination_count / n_evaluated, 4) if n_evaluated else 0.0

    # Per-regulation version fingerprint: hash the article corpus used at
    # evaluation time. Lets drift / changelog reason about staleness later.
    regulation_versions: dict[str, str] = {}
    for reg in regulation_scope:
        meta_map = _load_article_meta(reg)
        if meta_map:
            import hashlib as _hashlib
            corpus_blob = json.dumps(
                sorted(
                    (aid, a.get("content", "")) for aid, a in meta_map.items()
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            regulation_versions[reg] = _hashlib.sha256(corpus_blob).hexdigest()

    return {
        "doc_id": doc_id,
        "doc_filename": doc_filename,
        "doc_path": doc_path,
        "doc_sha256": doc_sha256,
        "doc_type": doc_type,
        "regulations": regulation_scope,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "articles_evaluated": n_evaluated,
        "violations": violations,
        "hallucination_flags": hallucination_count,
        "hallucination_rate": hallucination_rate,
        "generated_at": generated_at,
        "model": os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-8B"),
        "thinking_enabled": thinking_enabled,
        "regulation_versions": regulation_versions,
    }


# ---------------------------------------------------------------------------
# POA&M rendering
# ---------------------------------------------------------------------------

def _render_poam(
    violation_report: dict,
    debate_records: list[dict],
    state: dict,
    doc_id: str,
) -> POAMReport:
    """Render Jinja templates and write PDFs to outputs/reports/{doc_id}/POA&M/."""
    from backend.reports.assessment import render_assessment_report
    from backend.reports.remediation import render_remediation_report
    from backend.reports.pdf_renderer import markdown_to_pdf

    assessment_dir = _PROJECT_ROOT / "outputs" / "reports" / doc_id / "POA&M"
    os.makedirs(assessment_dir, exist_ok=True)

    assessment_path = str(assessment_dir / "assessment_report.pdf")
    remediation_path = str(assessment_dir / "remediation_report.pdf")

    assessment_md = render_assessment_report(violation_report, debate_records, state)
    markdown_to_pdf(assessment_md, assessment_path)

    remediation_md = render_remediation_report(violation_report, state)
    markdown_to_pdf(remediation_md, remediation_path)

    # Also drop the raw markdown alongside the PDF so reviewers can diff.
    (assessment_dir / "assessment_report.md").write_text(assessment_md, encoding="utf-8")
    (assessment_dir / "remediation_report.md").write_text(remediation_md, encoding="utf-8")

    return POAMReport(
        assessment_report_path=assessment_path,
        remediation_report_path=remediation_path,
    )


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def reporter_compute_node(state: dict) -> dict:
    """Phase 1 — compute risk, remediation, and the violation_report dict.

    Writes ``outputs/reports/{doc_id}/raw/violation_report.json`` so that
    drift detection (running next) can read it back and compare against
    the baseline report on disk.
    """
    debate_records: list[dict] = state.get("debate_records", [])
    doc_id: str = state.get("doc_id", "unknown")
    doc_filename: str = state.get("doc_filename", "") or doc_id
    doc_path: str = state.get("doc_path", "")
    doc_sha256: str = state.get("doc_sha256", "")
    doc_type: str = state.get("doc_type", "unknown")
    regulation_scope: list[str] = state.get("regulation_scope", [])
    thinking_enabled: bool = state.get("thinking_enabled", True)

    generated_at = datetime.now(timezone.utc).isoformat()

    canonical_records = deduplicate_records(debate_records)
    risk_score, risk_level = compute_risk_score(canonical_records)

    violations_needing_remediation = [
        rec for rec in canonical_records if rec["verdict"] in ("Missing", "Partial")
    ]

    remediation_map: dict[str, str] = {}
    remediation_thinking = ""
    remediation_raw = ""

    if violations_needing_remediation:
        try:
            remediation_map, remediation_thinking, remediation_raw = _generate_remediations(
                violations_needing_remediation,
                thinking=thinking_enabled,
            )
        except Exception as exc:
            for v in violations_needing_remediation:
                remediation_map[v["article_id"]] = (
                    f"Review and update the document to address {v['article_id']} "
                    f"({v['article_title']}). The current coverage is '{v['verdict']}'. "
                    f"Ensure all key requirements of this article are explicitly addressed."
                )
            remediation_thinking = ""
            remediation_raw = f"Error generating remediations: {exc}"

    violation_report = _build_violation_report(
        canonical_records=canonical_records,
        risk_score=risk_score,
        risk_level=risk_level,
        doc_id=doc_id,
        doc_filename=doc_filename,
        doc_path=doc_path,
        doc_sha256=doc_sha256,
        doc_type=doc_type,
        regulation_scope=regulation_scope,
        remediation_map=remediation_map,
        generated_at=generated_at,
        thinking_enabled=thinking_enabled,
    )

    # Persist the raw ViolationReport JSON (drift_node will read it back).
    raw_dir = _PROJECT_ROOT / "outputs" / "reports" / doc_id / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = raw_dir / "violation_report.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(violation_report, fh, indent=2, ensure_ascii=False)

    # Record the evaluation in the regulation changelog so subsequent watcher
    # runs can flag this document for re-evaluation when regulations change.
    try:
        from backend.regulation.changelog import regulation_changelog
        import hashlib as _hashlib

        for reg in regulation_scope:
            meta_map = _load_article_meta(reg)
            for aid, article in meta_map.items():
                content_hash = _hashlib.sha256(
                    article.get("content", "").encode("utf-8")
                ).hexdigest()
                regulation_changelog.record_evaluation(
                    doc_id=doc_id,
                    regulation=reg,
                    article_id=aid,
                    content_hash=content_hash,
                )
    except Exception:
        # Changelog is best-effort; never fail the pipeline because SQLite hiccuped.
        pass

    log_entry = make_log_entry(
        agent="reporter_compute",
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
        },
    )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "violation_report": violation_report,
        "pipeline_log": [log_entry],
    }


def reporter_render_node(state: dict) -> dict:
    """Phase 2 — render Jinja templates and write POA&M PDFs.

    Runs after drift detection so the PDFs include the latest drift tables.
    """
    doc_id: str = state.get("doc_id", "unknown")
    violation_report: dict = state.get("violation_report", {})
    debate_records: list[dict] = state.get("debate_records", [])

    poam = _render_poam(
        violation_report=violation_report,
        debate_records=debate_records,
        state=state,
        doc_id=doc_id,
    )

    log_entry = make_log_entry(
        agent="reporter_render",
        input_data={"doc_id": doc_id},
        raw_prompt=None,
        thinking_trace=None,
        raw_response=None,
        structured_output={
            "assessment_report_path": poam["assessment_report_path"],
            "remediation_report_path": poam["remediation_report_path"],
        },
    )

    return {
        "poam": poam,
        "pipeline_log": [log_entry],
    }


# ---------------------------------------------------------------------------
# Backwards compatibility: keep the old single-node entry point as an alias
# of reporter_compute_node + reporter_render_node, in case external code
# still references ``reporter_node``.
# ---------------------------------------------------------------------------

def reporter_node(state: dict) -> dict:
    """Legacy single-node reporter (kept for backward compatibility)."""
    compute_out = reporter_compute_node(state)
    merged_state = {**state, **compute_out}
    render_out = reporter_render_node(merged_state)
    merged_pipeline_log = compute_out.get("pipeline_log", []) + render_out.get("pipeline_log", [])
    return {
        **compute_out,
        **render_out,
        "pipeline_log": merged_pipeline_log,
    }


# Legacy export used by older imports / tests.
def generate_poam(violation_report, debate_records, state, doc_id):  # noqa: D401
    """Legacy helper preserved for back-compat — prefer reporter_render_node."""
    return _render_poam(violation_report, debate_records, state, doc_id)
