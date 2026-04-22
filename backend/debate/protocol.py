"""DebateProtocol — Advocate / Challenger / Arbiter adversarial debate round.

Orchestrates a single (chunk_text, clause) debate using three sequential
Qwen3-8B calls with thinking enabled, then applies a hallucination guard
before returning a fully populated DebateRecord.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from backend.agents.state import DebateRecord

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ADVOCATE_PROMPT = """\
You are a compliance advocate reviewing an enterprise policy against a regulatory requirement.
Your role: find every possible reading of the policy that satisfies this requirement.
Be thorough and generous in interpretation — find supporting language wherever it exists.

{regulation} REQUIREMENT:
{article_id} — {article_title}
{clause_text}

POLICY SECTION UNDER REVIEW:
{policy_chunk}

Find the strongest possible argument that this policy satisfies the requirement.
If you find supporting language, quote it exactly.

Respond in this exact JSON format:
{{"argument": "Your argument for compliance here", "cited_text": "exact verbatim quote from policy or null if none found", "confidence": 0.0}}"""

CHALLENGER_PROMPT = """\
You are a strict compliance auditor. A compliance advocate has argued:

ADVOCATE'S ARGUMENT:
{advocate_full_output}

Your role: challenge this argument. Find every gap, ambiguity, and omission that shows
this policy does NOT fully satisfy the regulatory requirement. You have seen the advocate's
best case — now find what they missed or overstated.

{regulation} REQUIREMENT:
{article_id} — {article_title}
{clause_text}

POLICY SECTION:
{policy_chunk}

Respond in this exact JSON format:
{{"counterargument": "Your challenge to the advocate's position", "gap_identified": "The specific element that is missing or insufficient", "confidence": 0.0}}"""

ARBITER_PROMPT = """\
You are the final compliance arbiter. You have heard both sides of a compliance debate.

ADVOCATE argued:
{advocate_output}

CHALLENGER argued:
{challenger_output}

{regulation} REQUIREMENT:
{article_id} — {article_title}
{clause_text}

POLICY SECTION:
{policy_chunk}

Weigh both arguments carefully. Consider: Does the policy actually satisfy the regulatory requirement?
The Advocate may have been too generous. The Challenger may have been too strict. Find the truth.

Coverage definitions:
- Full: policy explicitly and clearly addresses ALL key aspects of this requirement
- Partial: policy addresses some but not all aspects, or uses vague/ambiguous language
- Missing: policy does not address this requirement at all

Respond ONLY in this exact JSON format:
{{"coverage": "Full|Partial|Missing", "risk_level": "Critical|High|Medium|Low", "reasoning": "2-3 sentences referencing both the advocate and challenger arguments and the actual policy text", "cited_text": "exact verbatim quote from the policy that best satisfies the requirement, or null if nothing satisfies it", "debate_summary": "1 sentence on what the debate revealed about this policy"}}

CRITICAL: cited_text must be copied verbatim from the policy section above. If coverage is Full or Partial, cited_text must not be null."""


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def safe_parse_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output.

    Strategy:
    1. Try ``json.loads`` on the full text.
    2. Attempt to locate a JSON block (possibly inside markdown fences) and parse it.
    3. Fall back to regex key-value extraction.
    4. Return an empty dict as last resort.
    """
    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Look for a JSON object (possibly inside ```json ... ```)
    # Find the outermost { ... } pair
    patterns = [
        re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL),
        re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL),
        re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", re.DOTALL),
    ]
    for pat in patterns:
        match = pat.search(text)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                continue

    # 3. Regex key-value fallback for known keys
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
        # Match "key": "value" or "key": number
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

    if result:
        return result

    # 4. Empty dict as absolute fallback
    return {}


# ---------------------------------------------------------------------------
# Core debate round
# ---------------------------------------------------------------------------

def run_debate(
    chunk_text: str,
    clause: dict,
    chunk_index: int,
    qwen_runner,
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
    debate_max_new_tokens = int(os.environ.get("DEBATE_MAX_NEW_TOKENS", "256"))
    advocate_raw = qwen_runner.generate(
        advocate_prompt,
        thinking=True,
        max_new_tokens=debate_max_new_tokens,
    )
    advocate_parsed = safe_parse_json(advocate_raw["response"])

    advocate_argument = advocate_parsed.get("argument", advocate_raw["response"])
    advocate_cited_text = advocate_parsed.get("cited_text")
    advocate_confidence = float(advocate_parsed.get("confidence", 0.0))
    advocate_thinking = advocate_raw["thinking_trace"]

    # ── Step 2: Challenger ────────────────────────────────────────────────
    challenger_prompt = CHALLENGER_PROMPT.format(
        advocate_full_output=advocate_raw["full_output"],
        regulation=regulation,
        article_id=article_id,
        article_title=article_title,
        clause_text=clause_text,
        policy_chunk=chunk_text,
    )
    challenger_raw = qwen_runner.generate(
        challenger_prompt,
        thinking=True,
        max_new_tokens=debate_max_new_tokens,
    )
    challenger_parsed = safe_parse_json(challenger_raw["response"])

    challenger_argument = challenger_parsed.get(
        "counterargument", challenger_raw["response"]
    )
    challenger_gap = challenger_parsed.get("gap_identified", "")
    challenger_confidence = float(challenger_parsed.get("confidence", 0.0))
    challenger_thinking = challenger_raw["thinking_trace"]

    # ── Step 3: Arbiter ───────────────────────────────────────────────────
    arbiter_prompt = ARBITER_PROMPT.format(
        advocate_output=advocate_raw["response"],
        challenger_output=challenger_raw["response"],
        regulation=regulation,
        article_id=article_id,
        article_title=article_title,
        clause_text=clause_text,
        policy_chunk=chunk_text,
    )
    arbiter_raw = qwen_runner.generate(
        arbiter_prompt,
        thinking=True,
        max_new_tokens=debate_max_new_tokens,
    )
    arbiter_parsed = safe_parse_json(arbiter_raw["response"])

    verdict = arbiter_parsed.get("coverage", "Missing")
    # Normalize verdict to expected values
    if verdict not in ("Full", "Partial", "Missing"):
        verdict_lower = verdict.lower()
        if "full" in verdict_lower:
            verdict = "Full"
        elif "partial" in verdict_lower:
            verdict = "Partial"
        else:
            verdict = "Missing"

    _LEVELS = ("Critical", "High", "Medium", "Low")

    def _normalize_risk_level(raw: object, default: str | None) -> str | None:
        if raw is None or not isinstance(raw, str):
            return default
        t = raw.strip().title()
        if t in _LEVELS:
            return t
        low = raw.strip().lower()
        return {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}.get(low)

    risk_level = _normalize_risk_level(arbiter_parsed.get("risk_level"), None)
    if risk_level is None:
        # Fall back to regulatory article severity from RAG metadata (not a blanket "High").
        risk_level = _normalize_risk_level(clause.get("severity"), "Medium")
    if risk_level is None:
        risk_level = "Medium"

    reasoning = arbiter_parsed.get("reasoning", "")
    final_cited_text = arbiter_parsed.get("cited_text")
    debate_summary = arbiter_parsed.get("debate_summary", "")
    arbiter_thinking = arbiter_raw["thinking_trace"]

    # ── Hallucination guard ───────────────────────────────────────────────
    # If arbiter claims Full or Partial coverage with a cited_text, verify
    # that the citation actually appears verbatim in the policy chunk.
    hallucination_flag = False
    if verdict in ("Full", "Partial") and final_cited_text:
        if final_cited_text not in chunk_text:
            hallucination_flag = True
            # Downgrade Full to Partial when citation is fabricated
            if verdict == "Full":
                verdict = "Partial"

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
