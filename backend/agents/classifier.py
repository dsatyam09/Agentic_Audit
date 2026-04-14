"""ClassifierAgent: doc type + regulation routing.

Model: Qwen3-8B (local) — runs classification locally with zero API cost.
"""

import json
from backend.agents.state import ComplianceState
from backend.logging.pipeline_log import make_log_entry

# ── Regulation registry ──────────────────────────────────────────────────────

REGULATION_REGISTRY = {
    "gdpr": {
        "namespace": "gdpr",
        "status": "active",
        "focus_articles": ["art_5", "art_6", "art_7", "art_13", "art_14",
                           "art_17", "art_25", "art_32", "art_33", "art_44"],
    },
    "soc2": {
        "namespace": "soc2",
        "status": "active",
        "focus_articles": [],
    },
    "hipaa": {
        "namespace": "hipaa",
        "status": "active",
        "focus_articles": [],
    },
    "iso27001": {
        "namespace": "iso27001",
        "status": "planned",
        "focus_articles": [],
    },
}

# ── Cross-regulation exclusion matrix ────────────────────────────────────────

EXCLUDED_COMBINATIONS = [
    frozenset(["hipaa", "gdpr"]),
    frozenset(["hipaa", "soc2"]),
]

# Maps doc_type → preferred regulation when a conflict must be resolved
DOC_TYPE_PREFERENCE = {
    "privacy_policy": "gdpr",
    "data_handling": "gdpr",
    "security_sop": "soc2",
    "vendor_agreement": "hipaa",
    "breach_sop": "hipaa",
    "other": "gdpr",
}


def resolve_conflict(regulations: list[str], excluded: frozenset, doc_type: str) -> list[str]:
    """When an excluded pair is present, keep the more applicable regulation."""
    preferred = DOC_TYPE_PREFERENCE.get(doc_type, "gdpr")
    if preferred in excluded:
        keep = preferred
    else:
        keep = sorted(excluded)[0]
    return [r for r in regulations if r not in excluded or r == keep]


def enforce_exclusions(regulations: list[str], doc_type: str) -> list[str]:
    reg_set = set(regulations)
    for excluded in EXCLUDED_COMBINATIONS:
        if excluded.issubset(reg_set):
            regulations = resolve_conflict(regulations, excluded, doc_type)
            reg_set = set(regulations)
    return regulations


# ── Classifier prompt ────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """Classify this enterprise document and identify which compliance regulations apply.

Examples:
Document: "This Privacy Policy describes how Acme Corp collects and uses personal data of EU residents..."
→ {{"doc_type": "privacy_policy", "regulation_scope": ["gdpr"], "confidence": 0.95, "reasoning": "EU personal data processing — GDPR applies."}}

Document: "This Security SOP defines incident response and access control procedures..."
→ {{"doc_type": "security_sop", "regulation_scope": ["gdpr", "soc2"], "confidence": 0.85, "reasoning": "Security procedures relevant to GDPR Art. 32 and SOC 2 Security criteria."}}

Document: "This Business Associate Agreement covers PHI handling between covered entities..."
→ {{"doc_type": "vendor_agreement", "regulation_scope": ["hipaa"], "confidence": 0.92, "reasoning": "BAA covering PHI — HIPAA applies. Note: GDPR excluded due to retention policy conflict."}}

Classify:
{doc_snippet}
Filename: {doc_path}

Output ONLY JSON:
{{"doc_type": "privacy_policy|security_sop|vendor_agreement|data_handling|breach_sop|other", "regulation_scope": ["gdpr"|"soc2"|"hipaa"|"iso27001"], "confidence": 0.0-1.0, "reasoning": "1-2 sentences"}}"""


def classifier_node(state: ComplianceState) -> dict:
    """LangGraph node: classifies document type and regulation scope."""
    doc_snippet = state["doc_text"][:1500]
    doc_path = state["doc_path"]

    prompt = CLASSIFIER_PROMPT.format(doc_snippet=doc_snippet, doc_path=doc_path)

    from backend.debate.qwen_runner import qwen
    out = qwen.generate(prompt, thinking=False, max_new_tokens=256)
    raw_response = out["response"].strip()

    # Parse response
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        import re
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {
                "doc_type": "other",
                "regulation_scope": ["gdpr"],
                "confidence": 0.3,
                "reasoning": "Failed to parse classifier output, defaulting to GDPR.",
            }

    doc_type = result.get("doc_type", "other")
    regulation_scope = result.get("regulation_scope", ["gdpr"])
    confidence = result.get("confidence", 0.5)
    reasoning = result.get("reasoning", "")

    # Filter to active regulations only
    regulation_scope = [
        r for r in regulation_scope
        if r in REGULATION_REGISTRY and REGULATION_REGISTRY[r]["status"] == "active"
    ]
    if not regulation_scope:
        regulation_scope = ["gdpr"]

    # Enforce cross-regulation exclusions
    regulation_scope = enforce_exclusions(regulation_scope, doc_type)

    log = make_log_entry(
        agent="classifier",
        input_data=doc_snippet[:200],
        raw_prompt=prompt,
        thinking_trace=None,
        raw_response=raw_response,
        structured_output={
            "doc_type": doc_type,
            "regulation_scope": regulation_scope,
            "confidence": confidence,
            "reasoning": reasoning,
        },
    )

    return {
        "doc_type": doc_type,
        "regulation_scope": regulation_scope,
        "classifier_confidence": confidence,
        "classifier_reasoning": reasoning,
        "pipeline_log": [log],
    }
