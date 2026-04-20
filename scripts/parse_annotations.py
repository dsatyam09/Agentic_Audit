"""Extract ground-truth verdicts from annotation PDFs in test_datasets/.

For each doc, produces a mapping of {focus_article_id -> "Full"|"Partial"|"Missing"}.

Heuristics:
- "Fully Compliant" header => every focus article defaults to Full.
- Else scan findings. Article references are parsed from finding text:
    GDPR:  "Art. 13(1)(a)" -> "art_13"
    HIPAA: "45 CFR 164.520" -> "hipaa_164_520"
  Each referenced focus article receives the worst verdict among:
    [VIOLATION] -> Missing, [CONCERN] -> Partial, [COMPLIANT] -> Full.
- Focus articles never mentioned default to Full.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GDPR_FOCUS = ["art_5", "art_6", "art_7", "art_13", "art_14",
              "art_17", "art_25", "art_32", "art_33", "art_44"]
HIPAA_FOCUS = ["hipaa_164_306", "hipaa_164_308", "hipaa_164_310", "hipaa_164_312",
               "hipaa_164_316", "hipaa_164_404", "hipaa_164_408",
               "hipaa_164_502", "hipaa_164_524", "hipaa_164_530"]

FOCUS = {"gdpr": GDPR_FOCUS, "hipaa": HIPAA_FOCUS}

VERDICT_RANK = {"Full": 0, "Partial": 1, "Missing": 2}
RANK_VERDICT = {0: "Full", 1: "Partial", 2: "Missing"}

# Range references like "Art. 15-22" or "164.522-528" expand to every integer in
# the inclusive range — each one counted as a mention.
GDPR_ART_RANGE = re.compile(r"Art\.\s*(\d+)\s*[-–]\s*(\d+)")
GDPR_ART_SINGLE = re.compile(r"Art\.\s*(\d+)")
HIPAA_RANGE = re.compile(r"164\.(\d+)\s*[-–]\s*(\d+)")
HIPAA_SINGLE = re.compile(r"164\.(\d+)")

FINDING_RE = re.compile(
    r"Finding\s+\d+\s*:\s*\[(VIOLATION|CONCERN|COMPLIANT)\](.*?)(?=Finding\s+\d+\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)

TAG_VERDICT = {
    "VIOLATION": "Missing",
    "CONCERN": "Partial",
    "COMPLIANT": "Full",
}


def _pdf_text(path: Path) -> str:
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def _extract_gdpr_articles(body: str) -> list[str]:
    out: set[int] = set()
    for m in GDPR_ART_RANGE.finditer(body):
        a, b = int(m.group(1)), int(m.group(2))
        out.update(range(a, b + 1))
    remaining = GDPR_ART_RANGE.sub("", body)
    for m in GDPR_ART_SINGLE.finditer(remaining):
        out.add(int(m.group(1)))
    return [f"art_{n}" for n in sorted(out)]


def _extract_hipaa_articles(body: str) -> list[str]:
    out: set[int] = set()
    for m in HIPAA_RANGE.finditer(body):
        a, b = int(m.group(1)), int(m.group(2))
        out.update(range(a, b + 1))
    remaining = HIPAA_RANGE.sub("", body)
    for m in HIPAA_SINGLE.finditer(remaining):
        out.add(int(m.group(1)))
    return [f"hipaa_164_{n}" for n in sorted(out)]


def parse_annotation(pdf_path: Path, regulation: str) -> dict:
    text = _pdf_text(pdf_path)
    focus = FOCUS[regulation]
    verdicts: dict[str, str] = {a: "Full" for a in focus}

    header_match = re.search(r"Compliance Level:\s*([^\n]+)", text)
    level_txt = (header_match.group(1) if header_match else "").lower()
    fully_compliant = "fully compliant" in level_txt

    if fully_compliant:
        return {
            "pdf": pdf_path.name,
            "compliance_level": level_txt.strip(),
            "verdicts": verdicts,
        }

    extractor = _extract_gdpr_articles if regulation == "gdpr" else _extract_hipaa_articles
    for tag, body in FINDING_RE.findall(text):
        tag = tag.upper()
        verdict = TAG_VERDICT.get(tag, "Full")
        refs = extractor(body)
        for art in refs:
            if art not in verdicts:
                continue
            if VERDICT_RANK[verdict] > VERDICT_RANK[verdicts[art]]:
                verdicts[art] = verdict

    return {
        "pdf": pdf_path.name,
        "compliance_level": level_txt.strip(),
        "verdicts": verdicts,
    }


def build_ground_truth(regulation: str, prefer_source: str = "gpt") -> dict:
    """Build one ground-truth map per doc using the preferred annotator."""
    ann_dir = PROJECT_ROOT / "test_datasets" / regulation / "annotations"
    art_dir = PROJECT_ROOT / "test_datasets" / regulation / "articles"
    gt: dict[str, dict] = {}

    for article_pdf in sorted(art_dir.glob("*.pdf")):
        stem = article_pdf.stem
        sources_tried = []
        chosen = None
        for src in (prefer_source, "claude", "gemini", "gpt"):
            if src in sources_tried:
                continue
            sources_tried.append(src)
            candidate = ann_dir / f"{stem}_annotation_{src}.pdf"
            if candidate.exists():
                chosen = candidate
                break
        if chosen is None:
            continue
        parsed = parse_annotation(chosen, regulation)
        parsed["article_pdf"] = str(article_pdf)
        parsed["doc_stem"] = stem
        parsed["annotation_source"] = chosen.name
        gt[stem] = parsed
    return gt


def main():
    out_dir = PROJECT_ROOT / "data" / "testing" / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    for reg in ("gdpr", "hipaa"):
        gt = build_ground_truth(reg, prefer_source="gpt")
        out_path = out_dir / f"{reg}_annotations.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(gt, fh, indent=2)
        print(f"{reg}: {len(gt)} docs -> {out_path}")
        for stem, payload in gt.items():
            v = payload["verdicts"]
            counts = {lbl: sum(1 for x in v.values() if x == lbl)
                      for lbl in ("Full", "Partial", "Missing")}
            print(f"  {stem}: Full={counts['Full']} Partial={counts['Partial']} Missing={counts['Missing']}"
                  f"  ({payload['compliance_level']})")


if __name__ == "__main__":
    main()
