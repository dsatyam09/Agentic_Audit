"""DebateProtocol — Advocate / Challenger / Arbiter adversarial debate round.

Orchestrates a single (chunk_text, clause) debate using three sequential
Qwen3-8B calls with optional thinking, then applies hallucination and
schema guards before returning a fully populated DebateRecord.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.agents.state import DebateRecord

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
#
# We keep JSON schema instructions *separate* from the actual content so the
# model cannot echo a placeholder (e.g. "Your argument for compliance here")
# back as a genuine answer. The schema is described with <<angle brackets>>
# to make it clear these are slot names, not values. We also provide a
# tiny worked example so the model imitates structure rather than the slot
# string itself.
# ---------------------------------------------------------------------------

ADVOCATE_PROMPT = """\
ROLE: You are a compliance advocate reviewing an enterprise policy against a regulatory requirement.
GOAL: Find every supporting reading of the policy that satisfies this requirement. Be thorough.

{regulation} REQUIREMENT:
{article_id} — {article_title}
{clause_text}

POLICY SECTION UNDER REVIEW:
\"\"\"{policy_chunk}\"\"\"

TASK:
- Identify the strongest argument that this policy satisfies the requirement.
- If supporting language exists, copy the shortest verbatim span from the POLICY SECTION (not the requirement) that supports your argument.
- If no supporting language exists, set cited_text to null and argue accordingly.

RESPONSE FORMAT — respond with ONE JSON object and NOTHING else:
{{
  "argument": <string: 2-4 sentences of your reasoning, specific to this policy>,
  "cited_text": <string verbatim from POLICY SECTION, or null>,
  "confidence": <float between 0.0 and 1.0>
}}

Do not repeat the field descriptions. Replace every <...> slot with your own content."""

CHALLENGER_PROMPT = """\
ROLE: You are a strict compliance auditor. The advocate has argued for compliance.
GOAL: Find every gap, ambiguity, and omission that shows the policy does NOT fully satisfy the requirement.

ADVOCATE'S ARGUMENT:
{advocate_response}

{regulation} REQUIREMENT:
{article_id} — {article_title}
{clause_text}

POLICY SECTION:
\"\"\"{policy_chunk}\"\"\"

TASK:
- Identify at least one concrete gap in the policy with respect to the requirement.
- Your counterargument must reference specific language (or its absence) in the POLICY SECTION, not the advocate's rhetoric.

RESPONSE FORMAT — respond with ONE JSON object and NOTHING else:
{{
  "counterargument": <string: 2-4 sentences identifying weaknesses in the advocate's case>,
  "gap_identified": <string: the single most critical missing or insufficient element>,
  "confidence": <float between 0.0 and 1.0>
}}

Do not repeat the field descriptions. Replace every <...> slot with your own content."""

ARBITER_PROMPT = """\
ROLE: You are the final compliance arbiter. Weigh both sides.

ADVOCATE argued:
{advocate_response}

CHALLENGER argued:
{challenger_response}

{regulation} REQUIREMENT:
{article_id} — {article_title}
{clause_text}

POLICY SECTION:
\"\"\"{policy_chunk}\"\"\"

COVERAGE DEFINITIONS:
- Full: policy explicitly addresses ALL key aspects of this requirement.
- Partial: policy addresses some aspects, or uses vague/ambiguous language.
- Missing: policy does not address this requirement at all.

HARD RULES:
1. cited_text MUST be a verbatim substring of the POLICY SECTION (not the requirement text, not your reasoning, not the advocate/challenger output).
2. If coverage is Full or Partial, cited_text MUST be a non-empty verbatim quote from POLICY SECTION.
3. If coverage is Missing, cited_text MUST be null.

RESPONSE FORMAT — respond with ONE JSON object and NOTHING else:
{{
  "coverage": <one of: Full, Partial, Missing>,
  "risk_level": <one of: Critical, High, Medium, Low>,
  "reasoning": <string: 2-3 sentences referencing both sides and the actual policy text>,
  "cited_text": <string verbatim from POLICY SECTION, or null>,
  "debate_summary": <string: 1 sentence summarising what the debate revealed>
}}

Do not repeat the field descriptions. Replace every <...> slot with your own content."""


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

# Placeholder strings we treat as "not a real answer" so we don't let the
# model echo our template slots back into the report.
_TEMPLATE_PLACEHOLDERS = (
    "your argument",
    "your challenge",
    "your counterargument",
    "your reasoning",
    "your challenge to the advocate",
    "<string",
    "<float",
    "<one of",
    "verbatim from policy section",
    "2-4 sentences",
    "2-3 sentences",
)


def _looks_like_placeholder(value: Any) -> bool:
    """True when *value* is empty or matches a template slot description."""
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return True
    return any(p in normalized for p in _TEMPLATE_PLACEHOLDERS)


def _find_balanced_json(text: str) -> str | None:
    """Locate the first balanced ``{...}`` object in *text*.

    Uses a depth counter rather than regex, so it tolerates nested braces
    and strings containing braces. Returns ``None`` if no balanced object
    is found.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : i + 1]
    return None


def safe_parse_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output.

    Strategy:
    1. Try ``json.loads`` on the full text.
    2. Strip markdown fences and retry.
    3. Scan for a balanced ``{...}`` block and parse it.
    4. Regex key-value fallback for known schema keys.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    # 1. Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Markdown fence strip
    fence_patterns = [
        re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL),
        re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL),
    ]
    for pat in fence_patterns:
        match = pat.search(text)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                continue

    # 3. Balanced brace scan
    block = _find_balanced_json(text)
    if block:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Regex key-value fallback for known keys
    result: dict[str, Any] = {}
    for key in (
        "argument",
        "cited_text",
        "confidence",
        "counterargument",
        "gap_identified",
        "coverage",
        "risk_level",
        "reasoning",
        "debate_summary",
    ):
        m = re.search(
            rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*"|\d+\.?\d*|null|true|false)',
            text,
        )
        if m:
            raw = m.group(1)
            try:
                result[key] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                result[key] = raw.strip('"')

    return result


def _clean_string(value: Any, fallback: str = "") -> str:
    """Return a trimmed string, substituting *fallback* for placeholder text."""
    if not isinstance(value, str):
        return fallback
    if _looks_like_placeholder(value):
        return fallback
    return value.strip()


def _clean_optional_string(value: Any) -> str | None:
    """Return a trimmed string, or None when the value looks like a placeholder/null."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    if _looks_like_placeholder(value):
        return None
    stripped = value.strip()
    return stripped or None


# ---------------------------------------------------------------------------
# Core debate round
# ---------------------------------------------------------------------------

def run_debate(
    chunk_text: str,
    clause: dict,
    chunk_index: int,
    qwen_runner,
    thinking: bool = True,
) -> DebateRecord:
    """Execute a full Advocate -> Challenger -> Arbiter debate round.

    Parameters
    ----------
    chunk_text : str
        The policy text chunk being evaluated.
    clause : dict
        A RetrievedClause dict with keys: article_id, article_title,
        clause_text, severity, regulation, rerank_score.
    chunk_index : int
        Index of the chunk within the document.
    qwen_runner : QwenRunner
        The loaded Qwen3-8B inference wrapper.
    thinking : bool
        When False, disables ``<think>`` reasoning traces (C4-nothink ablation).

    Returns
    -------
    DebateRecord
        Fully populated debate record including hallucination guard results.
    """
    regulation = clause.get("regulation", "").upper()
    article_id = clause.get("article_id", "")
    article_title = clause.get("article_title", "")
    clause_text = clause.get("clause_text", "")

    # ── Step 1: Advocate ──────────────────────────────────────────────────
    advocate_prompt = ADVOCATE_PROMPT.format(
        regulation=regulation,
        article_id=article_id,
        article_title=article_title,
        clause_text=clause_text,
        policy_chunk=chunk_text,
    )
    advocate_raw = qwen_runner.generate(advocate_prompt, thinking=thinking)
    advocate_parsed = safe_parse_json(advocate_raw["response"])

    advocate_argument = _clean_string(
        advocate_parsed.get("argument"),
        fallback="Advocate did not return a parseable argument.",
    )
    advocate_cited_text = _clean_optional_string(advocate_parsed.get("cited_text"))
    if advocate_cited_text and advocate_cited_text not in chunk_text:
        advocate_cited_text = None
    try:
        advocate_confidence = float(advocate_parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        advocate_confidence = 0.0
    advocate_thinking = advocate_raw.get("thinking_trace", "")

    # ── Step 2: Challenger ────────────────────────────────────────────────
    # Pass the advocate's *parsed response* (post-</think>), not the full raw
    # output. This prevents the Challenger from seeing the <think> trace and
    # keeps it focused on the advocate's actual argument.
    challenger_prompt = CHALLENGER_PROMPT.format(
        advocate_response=advocate_raw["response"],
        regulation=regulation,
        article_id=article_id,
        article_title=article_title,
        clause_text=clause_text,
        policy_chunk=chunk_text,
    )
    challenger_raw = qwen_runner.generate(challenger_prompt, thinking=thinking)
    challenger_parsed = safe_parse_json(challenger_raw["response"])

    challenger_argument = _clean_string(
        challenger_parsed.get("counterargument"),
        fallback="Challenger did not return a parseable counterargument.",
    )
    challenger_gap = _clean_string(challenger_parsed.get("gap_identified"))
    try:
        challenger_confidence = float(challenger_parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        challenger_confidence = 0.0
    challenger_thinking = challenger_raw.get("thinking_trace", "")

    # ── Step 3: Arbiter ───────────────────────────────────────────────────
    arbiter_prompt = ARBITER_PROMPT.format(
        advocate_response=advocate_raw["response"],
        challenger_response=challenger_raw["response"],
        regulation=regulation,
        article_id=article_id,
        article_title=article_title,
        clause_text=clause_text,
        policy_chunk=chunk_text,
    )
    arbiter_raw = qwen_runner.generate(arbiter_prompt, thinking=thinking)
    arbiter_parsed = safe_parse_json(arbiter_raw["response"])

    verdict_raw = arbiter_parsed.get("coverage", "Missing")
    verdict = verdict_raw if isinstance(verdict_raw, str) else "Missing"
    if verdict not in ("Full", "Partial", "Missing"):
        vl = verdict.lower()
        if "full" in vl:
            verdict = "Full"
        elif "partial" in vl:
            verdict = "Partial"
        else:
            verdict = "Missing"

    risk_level = arbiter_parsed.get("risk_level", clause.get("severity", "High"))
    if risk_level not in ("Critical", "High", "Medium", "Low"):
        risk_level = clause.get("severity", "High") or "High"
        if risk_level not in ("Critical", "High", "Medium", "Low"):
            risk_level = "High"

    reasoning = _clean_string(
        arbiter_parsed.get("reasoning"),
        fallback="Arbiter reasoning was not parseable.",
    )
    final_cited_text = _clean_optional_string(arbiter_parsed.get("cited_text"))
    debate_summary = _clean_string(arbiter_parsed.get("debate_summary"))
    arbiter_thinking = arbiter_raw.get("thinking_trace", "")

    # ── Hallucination guard ───────────────────────────────────────────────
    # Rule 1: cited_text must be a verbatim substring of the policy chunk.
    hallucination_flag = False
    if final_cited_text and final_cited_text not in chunk_text:
        hallucination_flag = True
        final_cited_text = None  # strip fabricated quote so reports are honest
        if verdict == "Full":
            verdict = "Partial"

    # Rule 2: Full/Partial verdicts MUST have a non-null cited_text.
    # If the arbiter violated this contract, downgrade to Missing so the
    # report reflects reality (spec §5.3 / §7.1).
    if verdict in ("Full", "Partial") and not final_cited_text:
        verdict = "Missing"

    # Fallback debate_summary so downstream Jinja rendering is never empty.
    if not debate_summary:
        debate_summary = (
            f"Debate concluded with verdict: {verdict} "
            f"({'evidence cited' if final_cited_text else 'no verbatim evidence'})."
        )

    # ── Assemble DebateRecord ─────────────────────────────────────────────
    record: DebateRecord = {
        "article_id": article_id,
        "article_title": article_title,
        "regulation": clause.get("regulation", ""),
        "chunk_index": chunk_index,
        # Advocate
        "advocate_argument": advocate_argument,
        "advocate_cited_text": advocate_cited_text,
        "advocate_confidence": advocate_confidence,
        "advocate_thinking": advocate_thinking,
        # Challenger
        "challenger_argument": challenger_argument,
        "challenger_gap": challenger_gap,
        "challenger_confidence": challenger_confidence,
        "challenger_thinking": challenger_thinking,
        # Arbiter
        "verdict": verdict,
        "risk_level": risk_level,
        "reasoning": reasoning,
        "final_cited_text": final_cited_text,
        "debate_summary": debate_summary,
        "arbiter_thinking": arbiter_thinking,
        "hallucination_flag": hallucination_flag,
    }

    return record
