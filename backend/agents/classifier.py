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
        "focus_articles": [
            "art_5", "art_6", "art_7", "art_13", "art_14",
            "art_17", "art_25", "art_32", "art_33", "art_44",
        ],
    },
    "hipaa": {
        "namespace": "hipaa",
        "status": "active",
        # Key sections from HIPAA Security Rule (§164.3xx), Breach Notification (§164.4xx),
        # and Privacy Rule (§164.5xx) most commonly evaluated in enterprise audits.
        "focus_articles": [
            "hipaa_164_306",   # Security standards: General rules
            "hipaa_164_308",   # Administrative safeguards
            "hipaa_164_310",   # Physical safeguards
            "hipaa_164_312",   # Technical safeguards
            "hipaa_164_316",   # Policies and procedures
            "hipaa_164_404",   # Breach notification to individuals
            "hipaa_164_408",   # Notification to the Secretary
            "hipaa_164_502",   # Uses and disclosures of PHI: General rules
            "hipaa_164_524",   # Access of individuals to PHI
            "hipaa_164_530",   # Administrative requirements (Privacy Rule)
        ],
    },
    "nist": {
        "namespace": "nist",
        "status": "active",
        # Selected NIST SP 800-53 controls spanning the most-audited families:
        # AC (Access Control), AU (Audit), IA (Identity), IR (Incident Response),
        # RA (Risk), SC (System/Comms Protection), SI (System Integrity), CM, CP.
        "focus_articles": [
            "nist_ac-2",   # Account Management
            "nist_ac-3",   # Access Enforcement
            "nist_au-2",   # Event Logging
            "nist_ia-2",   # Identification and Authentication
            "nist_ir-4",   # Incident Handling
            "nist_ra-3",   # Risk Assessment
            "nist_sc-8",   # Transmission Confidentiality and Integrity
            "nist_si-2",   # Flaw Remediation
            "nist_cm-6",   # Configuration Settings
            "nist_cp-9",   # System Backup
        ],
    },
}

# ── Cross-regulation exclusion matrix ────────────────────────────────────────

EXCLUDED_COMBINATIONS = [
    # HIPAA (US health law) and GDPR (EU data law) have conflicting retention
    # and consent requirements — evaluating both on the same document gives
    # contradictory findings and confuses remediation.
    frozenset(["hipaa", "gdpr"]),
]

# Maps doc_type → preferred regulation when an exclusion conflict must be resolved
DOC_TYPE_PREFERENCE = {
    "privacy_policy": "gdpr",       # EU personal-data focus
    "data_handling": "gdpr",        # data processing focus
    "security_sop": "nist",         # security procedures → NIST SP 800-53
    "vendor_agreement": "hipaa",    # BAAs and PHI handling
    "breach_sop": "hipaa",          # breach notification requirements
    "other": "gdpr",                # safe default
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

Supported regulations: gdpr (EU personal data), hipaa (US health data / PHI), nist (security controls — NIST SP 800-53).
Rules:
- gdpr: EU personal data collection, processing, consent, data-subject rights, privacy policies.
- hipaa: US protected health information (PHI), covered entities, BAAs, breach notification.
- nist: Security SOPs, access control policies, incident response, system security plans.
- gdpr + nist can appear together (e.g., security SOP that also processes EU personal data).
- hipaa and gdpr are mutually exclusive (conflicting retention/consent rules).

Examples:
Document: "This Privacy Policy describes how Acme Corp collects and uses personal data of EU residents..."
→ {{"doc_type": "privacy_policy", "regulation_scope": ["gdpr"], "confidence": 0.95, "reasoning": "EU personal data processing — GDPR applies."}}

Document: "This Security SOP defines incident response and access control procedures for our systems..."
→ {{"doc_type": "security_sop", "regulation_scope": ["nist"], "confidence": 0.88, "reasoning": "Security procedures align with NIST SP 800-53 controls (IR-4, AC-2, AU-2)."}}

Document: "This Business Associate Agreement covers PHI handling between covered entities..."
→ {{"doc_type": "vendor_agreement", "regulation_scope": ["hipaa"], "confidence": 0.92, "reasoning": "BAA covering PHI — HIPAA applies. GDPR excluded due to conflict."}}

Document: "This Data Handling SOP governs EU employee data stored on our secure infrastructure..."
→ {{"doc_type": "data_handling", "regulation_scope": ["gdpr", "nist"], "confidence": 0.84, "reasoning": "EU personal data (GDPR) stored on auditable secure systems (NIST)."}}

Classify:
{doc_snippet}
Filename: {doc_path}

Output ONLY JSON:
{{"doc_type": "privacy_policy|security_sop|vendor_agreement|data_handling|breach_sop|other", "regulation_scope": ["gdpr"|"hipaa"|"nist"], "confidence": 0.0-1.0, "reasoning": "1-2 sentences"}}"""


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
