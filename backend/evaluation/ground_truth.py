"""Ground-truth resolver for uploaded documents.

Looks up annotation PDFs under ``test_datasets/<regulation>/annotations/``
that match an uploaded document by filename stem, parses them into
``{article_id: label}`` dicts and aggregates across multiple annotators
(Claude / Gemini / GPT) via majority vote with a *most-severe* tiebreaker.

Tag → label mapping used in annotation PDFs:

    [COMPLIANT]                                          -> Full
    [CONCERN]  [PARTIAL]                                 -> Partial
    [VIOLATION]  [NON-COMPLIANT]  [NONCOMPLIANT]  [MISSING] -> Missing
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEST_DATASETS = _PROJECT_ROOT / "test_datasets"

TAG_TO_LABEL = {
    "COMPLIANT": "Full",
    "FULL": "Full",
    "CONCERN": "Partial",
    "PARTIAL": "Partial",
    "VIOLATION": "Missing",
    "NON-COMPLIANT": "Missing",
    "NONCOMPLIANT": "Missing",
    "MISSING": "Missing",
}

# Severity ranking for tiebreakers (0 = least severe)
_SEVERITY = {"Full": 0, "Partial": 1, "Missing": 2}

# Article patterns per regulation
_GDPR_ART_RE = re.compile(r"\bArt\.?\s*(\d+)", re.IGNORECASE)
_HIPAA_CFR_RE = re.compile(r"\b(16[024])\.(\d+)\b")
# SOC2 Common Criteria: CC1.1 / C1.1 / A1.2 / P1.1 ...
_SOC2_RE = re.compile(r"\b(CC\d+|[ACPI]\d+)\.(\d+)\b", re.IGNORECASE)


def _detect_regulation(path: Path, text: str = "") -> str | None:
    """Infer the regulation from path components or content."""
    parts = {p.lower() for p in path.parts}
    for reg in ("gdpr", "hipaa", "soc2", "iso27001", "nist"):
        if reg in parts or any(reg in p for p in parts):
            return reg
    # Fallback: sniff from the first 500 chars of text
    head = text[:500].lower()
    if "gdpr" in head:
        return "gdpr"
    if "hipaa" in head or "45 cfr" in head:
        return "hipaa"
    if "soc 2" in head or "soc2" in head:
        return "soc2"
    return None


def _extract_article_ids(block: str, regulation: str) -> set[str]:
    """Extract normalized article_id values for a given regulation."""
    ids: set[str] = set()
    if regulation == "gdpr":
        for m in _GDPR_ART_RE.finditer(block):
            ids.add(f"art_{int(m.group(1))}")
    elif regulation == "hipaa":
        for m in _HIPAA_CFR_RE.finditer(block):
            ids.add(f"hipaa_{m.group(1)}_{m.group(2)}")
    elif regulation == "soc2":
        for m in _SOC2_RE.finditer(block):
            ids.add(f"soc2_{m.group(1).lower()}_{m.group(2)}")
    return ids


def parse_annotation_pdf(pdf_path: Path) -> dict[str, str]:
    """Parse a single annotation PDF into ``{article_id: label}``."""
    from backend.ingestion.parser import DocumentParser  # lazy import

    text = DocumentParser().parse(str(pdf_path))
    regulation = _detect_regulation(pdf_path, text)
    if not regulation:
        return {}

    # Split on "Finding N:" boundaries (the first chunk is the preamble)
    blocks = re.split(r"Finding\s+\d+\s*:", text)[1:]

    results: dict[str, str] = {}
    for block in blocks:
        tag_match = re.search(r"\[\s*([A-Za-z][A-Za-z\s\-/]+?)\s*\]", block)
        if not tag_match:
            continue
        raw_tag = tag_match.group(1).strip().upper().replace(" ", "").replace("/", "-")
        label = TAG_TO_LABEL.get(raw_tag)
        if not label:
            continue

        articles = _extract_article_ids(block, regulation)
        for art_id in articles:
            # Keep the most-severe label if the same article appears in
            # multiple findings within the same annotator.
            if art_id not in results or _SEVERITY[label] > _SEVERITY[results[art_id]]:
                results[art_id] = label

    return results


def _normalize_stem(name: str) -> str:
    """Lowercase stem without extension and without trailing ``_annotation_*``."""
    stem = Path(name).stem.lower()
    # Strip a trailing `_annotation_<annotator>` if present
    return re.sub(r"_annotation_[a-z0-9]+$", "", stem)


def find_annotation_pdfs(uploaded_filename: str) -> list[Path]:
    """Locate annotation PDFs that match an uploaded document by filename stem.

    Strategy (in order):

    1. Exact match: ``<stem>_annotation_*.pdf`` under any ``test_datasets/*/annotations/``.
    2. Loose match: any annotation PDF whose normalized stem equals the uploaded stem.
    3. Distinctive-token match: any annotation PDF whose stem contains the most
       distinctive token from the uploaded stem (e.g. "streamvibe").
    """
    if not _TEST_DATASETS.exists():
        return []

    uploaded_stem = Path(uploaded_filename).stem.lower()
    all_annotations = list(_TEST_DATASETS.glob("*/annotations/*.pdf"))

    # 1. Strict: uploaded_stem is the exact prefix
    strict = [
        p for p in all_annotations
        if p.stem.lower().startswith(f"{uploaded_stem}_annotation_")
    ]
    if strict:
        return strict

    # 2. Loose: normalized annotation stem == uploaded stem
    loose = [p for p in all_annotations if _normalize_stem(p.name) == uploaded_stem]
    if loose:
        return loose

    # 3. Distinctive-token: look for the "company" identifier in the uploaded stem.
    #    Skip common prefix tokens used in the corpus (gdpr/hipaa/soc2/compliant/...).
    skip = {
        "gdpr", "hipaa", "soc2", "iso27001", "nist",
        "compliant", "partial", "known", "tricky", "noncompliant", "missing",
        "policy", "privacy", "notice", "doc", "document", "pdf",
    }
    tokens = [t for t in re.split(r"[^a-z0-9]+", uploaded_stem) if t and t not in skip]
    # Prefer the longest distinctive token
    tokens.sort(key=len, reverse=True)
    for tok in tokens:
        if len(tok) < 4:
            continue
        matches = [p for p in all_annotations if tok in p.stem.lower()]
        if matches:
            return matches

    return []


def _aggregate_annotations(per_annotator: dict[str, dict[str, str]]) -> dict[str, str]:
    """Aggregate multiple annotators' labels via majority vote.

    Tiebreaker: most-severe label wins (Missing > Partial > Full).  This is
    the conservative choice appropriate for compliance auditing — when
    annotators disagree, the stricter label is preferred.
    """
    all_articles: set[str] = set()
    for labels in per_annotator.values():
        all_articles.update(labels.keys())

    out: dict[str, str] = {}
    for art_id in all_articles:
        votes = [d[art_id] for d in per_annotator.values() if art_id in d]
        if not votes:
            continue
        counts = Counter(votes).most_common()
        top_count = counts[0][1]
        top_labels = [lbl for lbl, c in counts if c == top_count]
        if len(top_labels) == 1:
            out[art_id] = top_labels[0]
        else:
            top_labels.sort(key=lambda x: _SEVERITY[x], reverse=True)
            out[art_id] = top_labels[0]
    return out


def _annotator_name_from_path(path: Path) -> str:
    """Extract the annotator label from a filename like
    ``gdpr_compliant_streamvibe_annotation_claude.pdf``."""
    m = re.search(r"_annotation_([a-z0-9]+)$", path.stem.lower())
    return m.group(1) if m else path.stem


def resolve_ground_truth(
    uploaded_filename: str,
    annotation_paths: Iterable[Path] | None = None,
) -> tuple[dict[str, str], dict]:
    """Resolve per-article ground-truth labels for an uploaded document.

    Returns a ``(labels, source_info)`` tuple:

    * ``labels``       — flat ``{article_id: label}`` (possibly empty).
    * ``source_info``  — metadata describing what was used:
        ``{ "source": "auto"|"none", "matched_stem": ..., "annotators": [...],
            "per_annotator": {"claude": {...}, ...} }``
    """
    paths = list(annotation_paths) if annotation_paths is not None else find_annotation_pdfs(uploaded_filename)
    source_info: dict = {
        "source": "none",
        "matched_stem": None,
        "annotators": [],
        "per_annotator": {},
        "files": [],
    }

    if not paths:
        return {}, source_info

    per_annotator: dict[str, dict[str, str]] = {}
    for p in paths:
        annot = _annotator_name_from_path(p)
        parsed = parse_annotation_pdf(p)
        if parsed:
            per_annotator[annot] = parsed

    if not per_annotator:
        return {}, source_info

    aggregated = _aggregate_annotations(per_annotator)

    # Pick an informative "matched_stem": the normalized stem of the first annotation
    source_info.update(
        source="auto",
        matched_stem=_normalize_stem(paths[0].name),
        annotators=sorted(per_annotator.keys()),
        per_annotator={k: v for k, v in per_annotator.items()},
        files=[str(p.relative_to(_PROJECT_ROOT)) for p in paths],
    )
    return aggregated, source_info
