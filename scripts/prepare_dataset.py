"""Filter + enrich raw regulation JSON → articles.json.

Usage:
    python scripts/prepare_dataset.py --regulation gdpr
    python scripts/prepare_dataset.py --regulation hipaa
    python scripts/prepare_dataset.py --regulation nist
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Focus articles and severity maps per regulation ──────────────────────────

GDPR_FOCUS_ARTICLES = {
    "art_5": {"severity": "Critical", "key_requirements": [
        "lawfulness, fairness, transparency",
        "purpose limitation stated",
        "data minimisation",
        "accuracy obligation",
        "storage limitation with retention period",
        "integrity and confidentiality measures",
    ]},
    "art_6": {"severity": "Critical", "key_requirements": [
        "at least one lawful basis identified",
        "consent OR contract OR legal obligation OR vital interest OR public task OR legitimate interest",
        "lawful basis clearly stated for each processing purpose",
    ]},
    "art_7": {"severity": "High", "key_requirements": [
        "consent freely given",
        "consent specific and informed",
        "right to withdraw consent mentioned",
        "withdrawal as easy as giving consent",
    ]},
    "art_13": {"severity": "High", "key_requirements": [
        "controller identity and contact details",
        "DPO contact details",
        "purposes and legal basis stated",
        "recipients or categories of recipients",
        "retention period or criteria",
        "data subject rights listed",
    ]},
    "art_14": {"severity": "High", "key_requirements": [
        "indirect collection notice",
        "categories of personal data",
        "source of data disclosed",
        "provision timeline stated",
    ]},
    "art_17": {"severity": "Critical", "key_requirements": [
        "right to request erasure stated",
        "erasure without undue delay",
        "grounds for erasure listed",
        "exceptions to erasure documented",
        "third-party notification on erasure",
    ]},
    "art_25": {"severity": "High", "key_requirements": [
        "data protection by design",
        "data protection by default",
        "technical measures described",
        "organisational measures described",
    ]},
    "art_32": {"severity": "Critical", "key_requirements": [
        "pseudonymisation and encryption mentioned",
        "confidentiality, integrity, availability, resilience",
        "ability to restore access after incident",
        "regular testing and evaluation process",
    ]},
    "art_33": {"severity": "High", "key_requirements": [
        "72-hour breach notification to supervisory authority",
        "breach notification process described",
        "documentation of breaches",
        "exception criteria stated",
    ]},
    "art_44": {"severity": "Medium", "key_requirements": [
        "international transfer conditions stated",
        "adequacy decision or appropriate safeguards",
        "standard contractual clauses or BCRs",
    ]},
}

HIPAA_FOCUS_ARTICLES = {
    "164_308_a1": {"severity": "Critical", "key_requirements": [
        "security management process",
        "risk analysis conducted",
        "risk management measures implemented",
        "sanction policy for violations",
    ]},
    "164_308_a3": {"severity": "High", "key_requirements": [
        "workforce security procedures",
        "authorization and supervision",
        "workforce clearance procedure",
        "termination procedures",
    ]},
    "164_308_a4": {"severity": "High", "key_requirements": [
        "information access management",
        "access authorization policies",
        "access establishment and modification",
    ]},
    "164_308_a5": {"severity": "High", "key_requirements": [
        "security awareness training",
        "security reminders",
        "protection from malicious software",
        "log-in monitoring",
        "password management",
    ]},
    "164_310_a1": {"severity": "Critical", "key_requirements": [
        "facility access controls",
        "contingency operations",
        "facility security plan",
        "access control and validation procedures",
    ]},
    "164_310_d1": {"severity": "High", "key_requirements": [
        "device and media controls",
        "disposal procedures",
        "media re-use procedures",
        "accountability for hardware and media",
    ]},
    "164_312_a1": {"severity": "Critical", "key_requirements": [
        "access controls implemented",
        "unique user identification",
        "emergency access procedure",
        "automatic logoff",
        "encryption and decryption",
    ]},
    "164_312_b": {"severity": "High", "key_requirements": [
        "audit controls implemented",
        "audit log review procedures",
    ]},
    "164_312_c1": {"severity": "High", "key_requirements": [
        "integrity controls",
        "mechanism to authenticate ePHI",
    ]},
    "164_312_e1": {"severity": "Critical", "key_requirements": [
        "transmission security",
        "integrity controls for transmission",
        "encryption for transmission",
    ]},
}

REGULATION_CONFIG = {
    "gdpr": {
        "focus_articles": GDPR_FOCUS_ARTICLES,
        "raw_file": "data/compliance/gdpr/gdpr_raw.json",
        "output_file": "data/compliance/gdpr/gdpr_articles.json",
        "id_prefix": "art_",
    },
    "hipaa": {
        "focus_articles": HIPAA_FOCUS_ARTICLES,
        "raw_file": "data/compliance/hipaa/hipaa_raw.json",
        "output_file": "data/compliance/hipaa/hipaa_articles.json",
        "id_prefix": "",
    },
}


def normalize_article_id(raw_id: str, regulation: str) -> str:
    """Normalize article ID from raw format to internal format."""
    raw_id = raw_id.strip().lower()
    if regulation == "gdpr":
        # "Art 17" or "Art. 17" → "art_17"
        raw_id = raw_id.replace("art.", "art").replace("art ", "art_").replace(" ", "_")
        if not raw_id.startswith("art_"):
            raw_id = "art_" + raw_id
    return raw_id


def prepare_regulation(regulation: str) -> list[dict]:
    """Filter raw regulation data to focus articles and enrich with severity + key_requirements."""
    config = REGULATION_CONFIG[regulation]
    focus = config["focus_articles"]

    raw_path = PROJECT_ROOT / config["raw_file"]
    if not raw_path.exists():
        print(f"Warning: Raw file not found at {raw_path}. Creating from existing data...")
        # Try to find data in the compliance_data directory (legacy location)
        legacy_path = PROJECT_ROOT / "compliance_data" / regulation / f"{regulation}_articles.json"
        if legacy_path.exists():
            print(f"  Found legacy data at {legacy_path}")
            return _prepare_from_legacy(regulation, legacy_path, focus)
        print(f"  No data found for {regulation}. Skipping.")
        return []

    with open(raw_path) as f:
        raw_data = json.load(f)

    # Build lookup for recitals
    recitals = {r["id"]: r for r in raw_data if r.get("type") == "Recital"}

    articles = []
    for item in raw_data:
        if item.get("type") != "Article":
            continue

        norm_id = normalize_article_id(item["id"], regulation)
        if norm_id not in focus:
            continue

        # Build recital context
        recital_context = ""
        for rec_ref in item.get("related_recitals", []):
            # Try to find recital by title match
            for rec_id, rec in recitals.items():
                if rec_ref in rec.get("title", ""):
                    recital_context += rec["content"][:300] + " "
                    break

        article = {
            "article_id": norm_id,
            "article_number": _extract_number(item["id"]),
            "article_title": item.get("title", ""),
            "regulation": regulation,
            "severity": focus[norm_id]["severity"],
            "key_requirements": focus[norm_id]["key_requirements"],
            "content": item.get("content", ""),
            "recital_context": recital_context.strip()[:600] if recital_context else "",
            "source_url": item.get("url", ""),
            "word_count": len(item.get("content", "").split()),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        articles.append(article)

    return articles


def _prepare_from_legacy(regulation: str, legacy_path: Path, focus: dict) -> list[dict]:
    """Prepare articles from legacy compliance_data format."""
    with open(legacy_path) as f:
        legacy_data = json.load(f)

    articles = []
    for item in legacy_data:
        norm_id = item.get("article_id", "")
        if norm_id not in focus:
            continue

        article = {
            "article_id": norm_id,
            "article_number": item.get("article_number", 0),
            "article_title": item.get("article_title", ""),
            "regulation": regulation,
            "severity": focus[norm_id]["severity"],
            "key_requirements": focus[norm_id]["key_requirements"],
            "content": item.get("content", ""),
            "recital_context": "",
            "source_url": item.get("url", ""),
            "word_count": item.get("word_count", len(item.get("content", "").split())),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        articles.append(article)

    return articles


def _extract_number(raw_id: str) -> int:
    """Extract numeric portion from an article ID."""
    import re
    match = re.search(r'(\d+)', raw_id)
    return int(match.group(1)) if match else 0


def main():
    parser = argparse.ArgumentParser(description="Prepare regulation dataset")
    parser.add_argument("--regulation", required=True, choices=list(REGULATION_CONFIG.keys()))
    args = parser.parse_args()

    regulation = args.regulation
    config = REGULATION_CONFIG[regulation]

    articles = prepare_regulation(regulation)

    if not articles:
        print(f"No articles prepared for {regulation}.")
        return

    output_path = PROJECT_ROOT / config["output_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(articles, f, indent=2)

    print(f"Prepared {len(articles)} articles for {regulation} → {output_path}")
    for a in articles:
        print(f"  {a['article_id']}: {a['article_title']} [{a['severity']}]")


if __name__ == "__main__":
    main()
