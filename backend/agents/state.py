"""ComplianceState TypedDict — the single most important file in the codebase.

Every agent reads from and writes to this state object.
Do not change this schema without updating the spec document.
"""

from typing import TypedDict, Annotated
import operator


class DebateRecord(TypedDict):
    """Result of one full Advocate→Challenger→Arbiter debate round for a single (chunk, clause) pair."""
    article_id: str
    article_title: str
    regulation: str                     # "gdpr" | "hipaa" | "nist"
    chunk_index: int
    # Advocate output
    advocate_argument: str
    advocate_cited_text: str | None     # exact quote from policy the Advocate found
    advocate_confidence: float
    advocate_thinking: str              # Qwen3 <think> trace
    # Challenger output (sees Advocate's full response)
    challenger_argument: str
    challenger_gap: str                 # specific missing element identified
    challenger_confidence: float
    challenger_thinking: str
    # Arbiter final verdict (sees both)
    verdict: str                        # "Full" | "Partial" | "Missing"
    risk_level: str                     # "Critical" | "High" | "Medium" | "Low"
    reasoning: str                      # 2-3 sentences referencing both sides
    final_cited_text: str | None        # verified citation in policy
    debate_summary: str                 # 1 sentence: what the debate revealed
    arbiter_thinking: str
    hallucination_flag: bool            # True if cited_text claimed but not found verbatim


class POAMReport(TypedDict):
    """Paths to the two generated POA&M report files."""
    assessment_report_path: str         # outputs/reports/{doc_id}/POA&M/assessment_report.md
    remediation_report_path: str        # outputs/reports/{doc_id}/POA&M/remediation_report.md


class ComplianceState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    doc_id: str                         # sha256(doc_path)[:12] — short stable identifier
    doc_path: str                       # absolute filesystem path
    doc_filename: str                   # basename of doc_path (shown in reports)
    doc_sha256: str                     # sha256 of document *contents* (full hex)
    doc_text: str
    doc_chunks: list[dict]              # [{chunk_index, chunk_text, char_start, char_end}]

    # ── Run-mode flags ─────────────────────────────────────────────────────
    thinking_enabled: bool              # C4 vs C4-nothink ablation toggle

    # ── Classifier output ──────────────────────────────────────────────────
    doc_type: str                       # "privacy_policy"|"security_sop"|"vendor_agreement"|"data_handling"|"breach_sop"|"other"
    regulation_scope: list[str]         # ["gdpr"] or ["gdpr","nist"] — enforces exclusion matrix
    classifier_confidence: float
    classifier_reasoning: str

    # ── Retrieval output ────────────────────────────────────────────────────
    retrieved_clauses: list[dict]       # per chunk: [{chunk_index, chunk_text, clauses:[RetrievedClause]}]

    # ── Debate output ───────────────────────────────────────────────────────
    debate_records: list[DebateRecord]

    # ── Reporter output ─────────────────────────────────────────────────────
    risk_score: float                   # 0.0 (compliant) → 4.0 (non-compliant)
    risk_level: str                     # "Low"|"Medium"|"High"|"Critical"
    violation_report: dict              # full ViolationReport — see Section 8
    poam: POAMReport                    # paths to Assessment + Remediation reports

    # ── Drift (optional) ────────────────────────────────────────────────────
    previous_report_path: str | None    # path to prior violation_report.json
    drift_result: dict | None           # DriftResult — see Section 8

    # ── Pipeline log (auto-merged by LangGraph) ─────────────────────────────
    pipeline_log: Annotated[list[dict], operator.add]
