"""Emit the 6 presentation tables requested for the final report.

Three categories × two regulations:

    1. Retrieval Quality — GDPR
    2. Retrieval Quality — HIPAA
    3. Clause Evaluation — GDPR
    4. Clause Evaluation — HIPAA
    5. Report Quality — GDPR
    6. Report Quality — HIPAA

Each table has one row per document plus a regulation-level mean row, so the
reader can see per-doc detail and the aggregate in one place. Targets from the
rubric (RAGAS Faithfulness ≥ 0.85, Answer Relevance ≥ 0.80, Context Precision
≥ 0.75) are surfaced as column headers. The semantic judge is the primary
source; Qwen Debate numbers are omitted here — the Qwen-0.5B model systematically
downgrades every verdict to Missing and isn't diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POAM = PROJECT_ROOT / "outputs" / "POA&M"
SEM_PATH = POAM / "_semantic_evaluation.json"


def _fmt(v, spec=".3f"):
    if v is None:
        return "–"
    if isinstance(v, (int, float)):
        return f"{v:{spec}}"
    return str(v)


def _sem_docs(reg: str) -> list[dict]:
    sem = json.loads(SEM_PATH.read_text())
    return [json.loads((POAM / stem / "semantic_metrics.json").read_text())
            for stem in sem["docs"] if stem.startswith(reg + "_")]


def _reg_agg(reg: str) -> dict:
    sem = json.loads(SEM_PATH.read_text())
    return sem["per_regulation"][reg]


def _retrieval_table(reg: str) -> str:
    docs = _sem_docs(reg)
    agg = _reg_agg(reg)

    rows = [
        "| Document | RAGAS Faithfulness | Answer Relevance | Context Precision |",
        "|---|---|---|---|",
    ]
    for d in docs:
        rg = d["ragas_proxy"]
        rows.append(
            f"| `{d['doc_stem']}` | "
            f"{_fmt(rg['faithfulness'])} | "
            f"{_fmt(rg['answer_relevance'])} | "
            f"{_fmt(rg['context_precision'])} |"
        )
    mr = agg["ragas_proxy_mean"]
    rows.append(
        f"| **{reg.upper()} mean (n={agg['n_documents']})** | "
        f"**{_fmt(mr['faithfulness'])}** | "
        f"**{_fmt(mr['answer_relevance'])}** | "
        f"**{_fmt(mr['context_precision'])}** |"
    )
    return "\n".join(rows)


def _clause_table(reg: str) -> str:
    docs = _sem_docs(reg)
    agg = _reg_agg(reg)

    rows = [
        "| Document | Precision | Recall | Cohen's Kappa |",
        "|---|---|---|---|",
    ]
    for d in docs:
        mac = d["clause_evaluation"]["macro_3class"]
        rows.append(
            f"| `{d['doc_stem']}` | "
            f"{_fmt(mac['macro_precision'])} | "
            f"{_fmt(mac['macro_recall'])} | "
            f"{_fmt(mac['cohens_kappa_3class'])} |"
        )
    mac = agg["clause_evaluation"]["macro_3class"]
    rows.append(
        f"| **{reg.upper()} overall (n={agg['n_article_evaluations']} articles)** | "
        f"**{_fmt(mac['macro_precision'])}** | "
        f"**{_fmt(mac['macro_recall'])}** | "
        f"**{_fmt(mac['cohens_kappa_3class'])}** |"
    )
    return "\n".join(rows)


def _report_table(reg: str) -> str:
    docs = _sem_docs(reg)
    agg = _reg_agg(reg)

    rows = [
        "| Document | Likert Score (1-4) | Hallucination Rate |",
        "|---|---|---|",
    ]
    for d in docs:
        rows.append(
            f"| `{d['doc_stem']}` | "
            f"{d['report_quality']['likert_4']} | "
            f"{_fmt(d['hallucination_rate'])} |"
        )
    rows.append(
        f"| **{reg.upper()} mean (n={agg['n_documents']})** | "
        f"**{_fmt(agg['report_quality_mean_likert'], '.2f')}** | "
        f"**{_fmt(agg['hallucination_rate_mean'])}** |"
    )
    return "\n".join(rows)


def main():
    sections = [
        "# Agentic Compliance Audit — Evaluation Tables",
        "",
        "Six tables: **Retrieval Quality / Clause Evaluation / Report Quality** "
        "× **GDPR / HIPAA**. Each table has one row per document plus a "
        "regulation-level aggregate.",
        "",
        "Source: semantic-similarity judge "
        "(`scripts/evaluate_semantic.py`) against legal-expert ground truth in "
        "`test_datasets/{gdpr,hipaa}/annotations/`.",
        "",
        "---",
        "",
        "## 1. Retrieval Quality — GDPR",
        "",
        _retrieval_table("gdpr"),
        "",
        "## 2. Retrieval Quality — HIPAA",
        "",
        _retrieval_table("hipaa"),
        "",
        "## 3. Clause Evaluation — GDPR",
        "",
        _clause_table("gdpr"),
        "",
        "## 4. Clause Evaluation — HIPAA",
        "",
        _clause_table("hipaa"),
        "",
        "## 5. Report Quality — GDPR",
        "",
        _report_table("gdpr"),
        "",
        "## 6. Report Quality — HIPAA",
        "",
        _report_table("hipaa"),
        "",
    ]

    md_path = POAM / "EVALUATION_TABLES.md"
    md_path.write_text("\n".join(sections))
    print(f"Wrote {md_path}")

    # Render PDFs (portrait + landscape). Teammates pasting cells into slides
    # usually prefer the landscape variant because the row heights collapse.
    _render_pdfs(md_path)


def _render_pdfs(md_path: Path) -> None:
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    import io
    import markdown as _markdown
    from xhtml2pdf import pisa
    from backend.reports.pdf_renderer import _CSS, markdown_to_pdf

    md_text = md_path.read_text()

    portrait_pdf = md_path.with_suffix(".pdf")
    markdown_to_pdf(md_text, str(portrait_pdf))
    print(f"Wrote {portrait_pdf} (A4 portrait)")

    import re as _re
    landscape_css = _re.sub(
        r"@page\s*\{[^}]*\}",
        "@page { size: A4 landscape; margin: 1.2cm 1.5cm 1.5cm 1.5cm; }",
        _CSS, count=1,
    )
    html = _markdown.markdown(md_text, extensions=["tables", "fenced_code", "nl2br"])
    full_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{landscape_css}</style></head><body>{html}</body></html>"
    )
    landscape_pdf = md_path.with_name(md_path.stem + "_landscape.pdf")
    with open(landscape_pdf, "wb") as fh:
        result = pisa.CreatePDF(io.StringIO(full_html), dest=fh, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"xhtml2pdf returned {result.err} errors")
    print(f"Wrote {landscape_pdf} (A4 landscape)")


if __name__ == "__main__":
    main()
