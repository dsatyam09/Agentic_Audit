# AI Compliance Monitoring Agent — Engineering Specification

> This document is the single source of truth for building this system.
> Every implementation decision is made here. Claude Code reads this and builds exactly what is described.
> Do not deviate from schemas, file paths, function signatures, or data contracts without updating this document first.

---

## 1. System Purpose & Novel Contributions

### What This System Does

An AI-powered compliance monitoring agent that evaluates enterprise documents (privacy policies, SOPs, vendor agreements, breach response procedures) against regulatory standards. It produces structured audit-ready reports in a POA&M (Plan of Action and Milestones) format — the same format used by real enterprise compliance teams.

The system continuously monitors for **compliance drift** as both regulations and internal documents evolve over time.

### Three Technical Novelties

**Novelty 1 — Adversarial Debate Evaluation**
Rather than a single LLM deciding compliance, three specialized agents debate each clause:
- `AdvocateAgent` — finds every reading of the policy that satisfies the requirement
- `ChallengerAgent` — sees the Advocate's argument, then finds every gap and omission
- `ArbiterAgent` — reads both arguments and delivers the final structured verdict

This is grounded in LLM debate research (Du et al., 2023 — *Improving Factuality via Multi-Agent Debate*) and maps directly to how legal compliance works in practice. No existing compliance RAG system does this.

**Novelty 2 — Local Open-Weight LLM with Visible Thinking Traces (Qwen3-8B)**
All debate agents run on `Qwen/Qwen3-8B` locally. Qwen3-8B produces `<think>...</think>` reasoning tokens before every answer. Every thinking trace is captured in the pipeline log store — making the system fully auditable and reproducible. This enables a unique ablation: thinking mode ON vs OFF.

**Novelty 3 — Regulation-Aware Continuous Update via Adaptive RAG Refresh**
Compliance regulations update frequently (e.g., GDPR amendments, new NIST guidelines). Rather than requiring manual re-indexing, the system implements an **Adaptive Regulation Refresh** mechanism:
- A scheduled `RegulationWatcher` polls official regulation sources (EUR-Lex for GDPR, HHS for HIPAA, etc.) for structural changes
- When a change is detected, it computes a semantic diff between old and new article embeddings
- Only articles with cosine distance > threshold get re-embedded and re-indexed
- A `RegulationChangelog` table tracks every update with timestamp and affected article IDs
- Active documents that were previously evaluated against the changed articles are flagged for re-evaluation
- This is novel: existing compliance systems treat regulation data as static; this one treats it as a live feed

**Novelty 4 — Compliance Drift with Semantic Regression Score (SRS)**
Compares two versions of the same enterprise document. Beyond flagging regressions, computes a continuous **Semantic Regression Score**:
```
SRS = (coverage_rank_drop × risk_weight) × (1 + cosine_distance(cited_text_v1, cited_text_v2))
```
A clause removal scores SRS ≈ 4.0. A subtle language weakening scores SRS ≈ 1.5. This prioritizes remediation effort.

---

## 2. Compliance Scope & Regulation Matrix

### Supported Regulations

| Regulation | Namespace | Status | Articles indexed | Focus Articles |
|---|---|---|---|---|
| GDPR (EU) | `gdpr` | Active | 10 | Art. 5, 6, 7, 13, 14, 17, 25, 32, 33, 44 |
| HIPAA (US Health) | `hipaa` | Active | 144 | §164.306/308/310/312/316/404/408/502/524/530 |
| NIST SP 800-53 | `nist` | Active | 324 | AC-2, AC-3, AU-2, IA-2, IR-4, RA-3, SC-8, SI-2, CM-6, CP-9 |

### Regulation Routing by Document Type

The `ClassifierAgent` maps each document type to its natural regulation(s):

| doc_type | Default regulation_scope | Rationale |
|---|---|---|
| `privacy_policy` | `["gdpr"]` | EU personal data collection and processing |
| `data_handling` | `["gdpr"]` | Data processing procedures |
| `security_sop` | `["nist"]` | Security controls → NIST SP 800-53 |
| `vendor_agreement` | `["hipaa"]` | Business Associate Agreements, PHI handling |
| `breach_sop` | `["hipaa"]` | Breach notification requirements |
| `other` | `["gdpr"]` | Safe default |

Multi-regulation is permitted when content genuinely spans domains
(e.g., a security SOP processing EU personal data → `["gdpr", "nist"]`).

### Cross-Regulation Exclusion

**`GDPR + HIPAA` are mutually exclusive** — GDPR mandates erasure on request while
HIPAA mandates minimum retention periods. Contradictory obligations cannot be
jointly evaluated on the same document. If both are assigned, the classifier
keeps only the one that best matches `doc_type`.

**NIST can coexist with either GDPR or HIPAA** — it is a security framework, not
a privacy/health law, so there is no conflict.

---

## 3. Repository Structure

`backend/` holds all Python logic and exposes a FastAPI layer at `backend/api/`. `frontend/` is empty and reserved for the teammate building the React/Next.js UI. Everything below reflects the actual on-disk layout.

```
Agentic_Audit/
│
├── spec.md                          ← this file — the single source of truth
├── README.md                        ← brief setup instructions
├── architecture.md                  ← step-by-step system walkthrough
├── requirements.txt
├── .env                             ← never commit (QWEN_MODEL_ID, AGENTIC_AUDIT_CORS_ORIGINS, …)
├── run_pipeline.py                  ← CLI: python run_pipeline.py --doc <file> [--no-thinking]
├── run_evaluation.py                ← CLI: legacy evaluation harness (5 conditions)
│
├── backend/                         ← all Python logic
│   ├── __init__.py
│   │
│   ├── api/                         ← FastAPI layer (consumed by frontend)
│   │   ├── main.py                  ← FastAPI app + CORS middleware
│   │   └── routes.py                ← /api/v1/{health,regulations,analyze,reports,...}
│   │
│   ├── agents/
│   │   ├── state.py                 ← ComplianceState TypedDict — DO NOT change without updating this spec
│   │   ├── classifier.py            ← doc_type + regulation routing + REGULATION_REGISTRY
│   │   ├── retrieval_agent.py       ← RAG node (chunk → top-k articles)
│   │   ├── debate_agent.py          ← orchestrates DebateProtocol over (chunk, article) pairs
│   │   └── reporter.py              ← split into reporter_compute_node + reporter_render_node
│   │
│   ├── debate/
│   │   ├── protocol.py              ← Advocate / Challenger / Arbiter prompts + run_debate()
│   │   └── qwen_runner.py           ← Qwen singleton (configurable via QWEN_MODEL_ID)
│   │
│   ├── graph.py                     ← LangGraph StateGraph + run_pipeline(doc_path, ...)
│   │
│   ├── ingestion/
│   │   ├── parser.py                ← PDF / DOCX / TXT → plain text (PyMuPDF, python-docx)
│   │   └── chunker.py               ← 512-token windows, 50-token overlap (tiktoken)
│   │
│   ├── retrieval/
│   │   ├── embedder.py              ← all-MiniLM-L6-v2 (384-dim) + SHA-256 cache
│   │   ├── vector_store.py          ← Chroma client, one collection per regulation namespace
│   │   └── reranker.py              ← cross-encoder/ms-marco-MiniLM-L-6-v2
│   │
│   ├── regulation/
│   │   ├── watcher.py               ← RegulationWatcher: polls EUR-Lex / eCFR / NIST CPRT
│   │   ├── differ.py                ← semantic diff between article versions
│   │   └── changelog.py             ← SQLite table tracking all updates
│   │
│   ├── reports/
│   │   ├── assessment.py            ← renders assessment_report.md via Jinja2
│   │   ├── remediation.py           ← renders remediation_report.md via Jinja2
│   │   ├── pdf_renderer.py          ← markdown_to_pdf() — Markdown → HTML → PDF (xhtml2pdf)
│   │   └── templates/
│   │       ├── assessment.md.jinja
│   │       └── remediation.md.jinja
│   │
│   ├── drift/
│   │   └── detector.py              ← Semantic Regression Score + drift_node
│   │
│   ├── logging/
│   │   └── pipeline_log.py          ← SQLite writer (outputs/pipeline_logs.db) + JSON sidecar
│   │
│   └── evaluation/
│       └── metrics.py               ← compute_metrics: Precision, Recall, F1, Cohen's Kappa
│
├── frontend/                        ← reserved for React/Next.js (see Section 19)
│   └── .gitkeep
│
├── data/
│   ├── compliance/                  ← regulation source data (indexed into Chroma)
│   │   ├── gdpr/
│   │   │   ├── gdpr_raw.json        ← 272 records (articles + recitals)
│   │   │   └── gdpr_articles.json   ← 10 focus articles, enriched
│   │   ├── hipaa/
│   │   │   └── hipaa_articles.json  ← 144 CFR sections (§160/§164)
│   │   ├── nist/
│   │   │   └── nist_articles.json   ← 324 SP 800-53 controls
│   │   ├── soc2/                    ← reserved (placeholder, not yet indexed)
│   │   └── iso27001/                ← reserved (placeholder, not yet indexed)
│   │
│   └── chroma_db/                   ← Chroma persistent vector store (generated, gitignored)
│
├── test_datasets/                   ← evaluation inputs + ground-truth annotation PDFs
│   ├── gdpr/
│   │   ├── articles/                ← gdpr_compliant_streamvibe.pdf, gdpr_partial_known_quickdeals.pdf, …
│   │   └── annotations/             ← {doc_stem}_annotation_{claude|gpt|gemini}.pdf
│   ├── hipaa/
│   │   ├── articles/                ← hipaa_compliant_clearwater.pdf, …
│   │   └── annotations/
│   ├── soc2/                        ← reserved (placeholder)
│   ├── gdpr_soc2_combined/          ← reserved
│   └── hipaa_soc2_combined/         ← reserved
│
├── outputs/                         ← all generated outputs (gitignored)
│   ├── reports/
│   │   └── {doc_id}/                ← doc_id = sha256(abs_doc_path)[:12]
│   │       ├── POA&M/
│   │       │   ├── assessment_report.pdf
│   │       │   └── remediation_report.pdf
│   │       └── raw/
│   │           └── violation_report.json
│   ├── logs/
│   │   └── {doc_id}_{run_id}.json   ← per-run pipeline log with thinking traces
│   ├── pipeline_logs.db             ← SQLite, all runs aggregated
│   ├── drift/
│   │   └── {doc_id}_drift_{ts}.json
│   └── POA&M/                       ← evaluation harness outputs (per-doc + aggregate)
│       ├── _semantic_evaluation.json
│       ├── _evaluation_report.json
│       ├── EVALUATION_TABLES.md
│       ├── EVALUATION_TABLES.pdf
│       ├── EVALUATION_TABLES_landscape.pdf
│       └── {doc_stem}/
│           ├── violation_report.json
│           ├── metrics.json
│           ├── semantic_metrics.json
│           ├── assessment_report.pdf
│           └── remediation_report.pdf
│
└── scripts/
    ├── prepare_dataset.py           ← raw regulation JSON → enriched articles.json
    ├── index_regulations.py         ← embed articles.json → Chroma (run once per regulation)
    ├── generate_docs.py             ← synthetic enterprise document generator
    ├── annotate_ground_truth.py     ← CLI annotation helper
    ├── parse_annotations.py         ← extract Full/Partial/Missing labels from annotation PDFs
    ├── batch_evaluate.py            ← run pipeline across every doc in test_datasets/
    ├── compute_full_metrics.py      ← attach RAGAS proxies + classification metrics post-hoc
    ├── enrich_metrics.py            ← recompute retrieval recall from pipeline_logs.db
    ├── evaluate_semantic.py         ← deterministic semantic-similarity judge (PRIMARY evaluator)
    └── generate_tables.py           ← emits EVALUATION_TABLES.{md,pdf} (6-table presentation report)
```

---

## 4. Dataset Specification

### 4.1 Regulation Data Format

All regulation data follows this schema regardless of source regulation. This is the format `gdpr_raw.json` already uses and the format all future regulation files must match.

**Raw source file schema** (`data/compliance/{reg}/{reg}_raw.json`):
```json
[
  {
    "id": "Art 17",
    "type": "Article",
    "title": "Art. 17 GDPR Right to erasure ('right to be forgotten')",
    "content": "The data subject shall have the right to obtain from the controller...",
    "url": "https://gdpr-info.eu/art-17-gdpr/",
    "related_recitals": ["(65) Right to Erasure", "(66) Right to Erasure"]
  },
  {
    "id": "Recital 65",
    "type": "Recital",
    "title": "Recital 65 Right to Erasure",
    "content": "A data subject should have the right to have personal data...",
    "url": "https://gdpr-info.eu/recitals/no-65/"
  }
]
```

**Enriched articles file schema** (`data/compliance/{reg}/{reg}_articles.json`) — generated by `scripts/prepare_dataset.py`:
```json
[
  {
    "article_id": "art_17",
    "article_number": 17,
    "article_title": "Art. 17 GDPR Right to erasure ('right to be forgotten')",
    "regulation": "gdpr",
    "severity": "Critical",
    "key_requirements": [
      "right to request erasure stated",
      "erasure without undue delay",
      "grounds for erasure listed",
      "exceptions to erasure documented",
      "third-party notification on erasure"
    ],
    "content": "The data subject shall have the right to obtain...",
    "recital_context": "Recital 65: A data subject should have the right... [first 300 chars]",
    "source_url": "https://gdpr-info.eu/art-17-gdpr/",
    "word_count": 390,
    "last_updated": "2025-04-01T00:00:00Z"
  }
]
```

### 4.2 GDPR Dataset (Current Status — Done)

- **Source:** `data/compliance/gdpr/gdpr_raw.json` — 272 records (99 articles + 173 recitals)
- **Enriched:** `data/compliance/gdpr/gdpr_articles.json` — 10 focus articles with severity + key_requirements
- **Chroma:** 10 vectors in namespace `"gdpr"` — one per article, recital context appended
- **Severity map:**

```python
GDPR_SEVERITY = {
    "art_5":  "Critical",   # Core principles
    "art_6":  "Critical",   # Lawful basis — most commonly missing
    "art_7":  "High",       # Consent conditions
    "art_13": "High",       # Direct collection notice
    "art_14": "High",       # Indirect collection notice
    "art_17": "Critical",   # Right to erasure
    "art_25": "High",       # Privacy by design
    "art_32": "Critical",   # Security measures
    "art_33": "High",       # 72-hour breach notification
    "art_44": "Medium",     # International transfers
}
```

### 4.3 Ground Truth Annotation Format

Ground truth lives as **annotation PDFs** (one per LLM annotator: Claude, GPT-4, Gemini) under `test_datasets/{reg}/annotations/`. These PDFs use a structured format — each finding tagged `[VIOLATION]`, `[CONCERN]`, or `[COMPLIANT]` with article references — and are parsed at evaluation time by `scripts/parse_annotations.py`.

**Layout:**
```
test_datasets/
├── gdpr/
│   ├── articles/
│   │   ├── gdpr_compliant_streamvibe.pdf
│   │   ├── gdpr_partial_known_quickdeals.pdf
│   │   └── gdpr_partial_tricky_novasphere.pdf
│   └── annotations/
│       ├── gdpr_compliant_streamvibe_annotation_claude.pdf
│       ├── gdpr_compliant_streamvibe_annotation_gpt.pdf
│       ├── gdpr_compliant_streamvibe_annotation_gemini.pdf
│       ├── gdpr_partial_known_quickdeals_annotation_claude.pdf
│       └── …
└── hipaa/  (same structure: hipaa_compliant_clearwater, hipaa_partial_known_sunrise, hipaa_partial_tricky_pinnacle)
```

**Parsed verdict map** (in-memory schema produced by `parse_annotations.py`):
```python
{
    "doc_stem": "gdpr_partial_known_quickdeals",
    "regulation": "gdpr",
    "verdicts": {
        "art_5":  "Partial",
        "art_6":  "Missing",
        "art_7":  "Missing",
        "art_13": "Full",
        "art_14": "Missing",
        "art_17": "Missing",
        "art_25": "Partial",
        "art_32": "Missing",
        "art_33": "Missing",
        "art_44": "Full",
    },
    "annotators": ["claude", "gpt4", "gemini"],
    "majority_verdict_used": True,
}
```

**Parsing rules** (`scripts/parse_annotations.py`):
- A "Fully Compliant" header on the annotation PDF → every focus article defaults to `Full`.
- Otherwise scan `Finding N: [TAG] …` blocks. `[VIOLATION]` → `Missing`, `[CONCERN]` → `Partial`, `[COMPLIANT]` → `Full`.
- Article references are extracted with regex: `Art. 13(1)(a)` → `art_13`; `45 CFR 164.520` → `hipaa_164_520`. Ranges (`Art. 15-22`) expand to every integer.
- Each focus article receives the **worst** verdict among its mentions; unmentioned focus articles default to `Full`.
- The three annotators are reconciled by majority vote across models (ties broken by worst verdict).

**Important constraints:**
- All testing documents are synthetic — generated to have known compliance characteristics.
- No HIPAA+GDPR combined documents exist (excluded by design — see Section 2).
- SOC 2 / ISO 27001 / multi-regulation document sets exist as empty placeholders awaiting future annotation.

### 4.4 Adding a New Regulation

When a new regulation dataset arrives (in the same JSON format as `gdpr_raw.json`):

```bash
# 1. Place raw file
cp [new_reg].json data/compliance/[reg]/[reg]_raw.json

# 2. Add FOCUS_ARTICLES and SEVERITY_MAP to scripts/prepare_dataset.py
# 3. Run enrichment
python scripts/prepare_dataset.py --regulation [reg]

# 4. Index into Chroma
python scripts/index_regulations.py --regulation [reg]
# or index all at once:
python scripts/index_regulations.py --all

# 5. Update REGULATION_REGISTRY in backend/agents/classifier.py
# 6. Add test documents to data/testing/documents/[reg]/
# 7. Add ground truth to data/testing/ground_truth/[reg]_annotations.json
```

---

## 5. LangGraph State Schema

**File: `backend/agents/state.py`**

This is the single most important file in the codebase. Every agent reads from and writes to this state object. Do not change this schema without updating this spec document.

```python
from typing import TypedDict, Annotated
import operator

class DebateRecord(TypedDict):
    """Result of one full Advocate→Challenger→Arbiter debate round for a single (chunk, clause) pair."""
    article_id:              str
    article_title:           str
    regulation:              str             # "gdpr" | "hipaa" | "nist"
    chunk_index:             int
    # Advocate output
    advocate_argument:       str
    advocate_cited_text:     str | None      # exact quote from policy the Advocate found
    advocate_confidence:     float
    advocate_thinking:       str             # Qwen3 <think> trace
    # Challenger output (sees Advocate's full response)
    challenger_argument:     str
    challenger_gap:          str             # specific missing element identified
    challenger_confidence:   float
    challenger_thinking:     str
    # Arbiter final verdict (sees both)
    verdict:                 str             # "Full" | "Partial" | "Missing"
    risk_level:              str             # "Critical" | "High" | "Medium" | "Low"
    reasoning:               str             # 2-3 sentences referencing both sides
    final_cited_text:        str | None      # verified citation in policy
    debate_summary:          str             # 1 sentence: what the debate revealed
    arbiter_thinking:        str
    hallucination_flag:      bool            # True if cited_text claimed but not found verbatim

class POAMReport(TypedDict):
    """Paths to the two generated POA&M PDF report files."""
    assessment_report_path:  str            # outputs/reports/{doc_id}/POA&M/assessment_report.pdf
    remediation_report_path: str            # outputs/reports/{doc_id}/POA&M/remediation_report.pdf

class ComplianceState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    doc_id:                  str
    doc_path:                str
    doc_text:                str
    doc_chunks:              list[dict]      # [{chunk_index, chunk_text, char_start, char_end}]

    # ── Classifier output ──────────────────────────────────────────────────
    doc_type:                str             # "privacy_policy"|"security_sop"|"vendor_agreement"|"data_handling"|"breach_sop"|"other"
    regulation_scope:        list[str]       # ["gdpr"] or ["gdpr","nist"] — enforces exclusion matrix
    classifier_confidence:   float
    classifier_reasoning:    str

    # ── Retrieval output ────────────────────────────────────────────────────
    retrieved_clauses:       list[dict]      # per chunk: [{chunk_index, chunk_text, clauses:[RetrievedClause]}]

    # ── Debate output ───────────────────────────────────────────────────────
    debate_records:          list[DebateRecord]

    # ── Reporter output ─────────────────────────────────────────────────────
    risk_score:              float           # 0.0 (compliant) → 4.0 (non-compliant)
    risk_level:              str             # "Low"|"Medium"|"High"|"Critical"
    violation_report:        dict            # full ViolationReport — see Section 8
    poam:                    POAMReport      # paths to Assessment + Remediation reports

    # ── Drift (optional) ────────────────────────────────────────────────────
    previous_report_path:    str | None      # path to prior violation_report.json
    drift_result:            dict | None     # DriftResult — see Section 8

    # ── Pipeline log (auto-merged by LangGraph) ─────────────────────────────
    pipeline_log: Annotated[list[dict], operator.add]
```

---

## 6. Agent Specifications

### 6.1 ClassifierAgent (`backend/agents/classifier.py`)

**Model:** Qwen (local) — same `QwenRunner` singleton used by debate agents. `thinking=False`, `max_new_tokens=256`.

**Inputs:** `state["doc_text"][:1500]`, `state["doc_path"]`

**Outputs written to state:** `doc_type`, `regulation_scope`, `classifier_confidence`, `classifier_reasoning`

**Key logic:**
1. Run few-shot classification prompt → parse JSON `{doc_type, regulation_scope, confidence, reasoning}`
2. Filter to active regulations only (inactive namespaces are silently dropped)
3. Apply exclusion matrix: if `regulation_scope` contains `["hipaa", "gdpr"]`, keep only the one that best matches `doc_type`
4. Log entry appended to `pipeline_log`

**Exclusion enforcement:**
```python
EXCLUDED_COMBINATIONS = [
    frozenset(["hipaa", "gdpr"]),   # conflicting retention/consent requirements
]

DOC_TYPE_PREFERENCE = {
    "privacy_policy": "gdpr",
    "data_handling":  "gdpr",
    "security_sop":   "nist",
    "vendor_agreement": "hipaa",
    "breach_sop":     "hipaa",
    "other":          "gdpr",
}

def enforce_exclusions(regulations: list[str], doc_type: str) -> list[str]:
    reg_set = set(regulations)
    for excluded in EXCLUDED_COMBINATIONS:
        if excluded.issubset(reg_set):
            regulations = resolve_conflict(regulations, excluded, doc_type)
            reg_set = set(regulations)
    return regulations
```

### 6.2 RetrievalAgent (`backend/agents/retrieval_agent.py`)

**No LLM call.** Pure vector search + reranking.

**Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) with SHA-256 in-memory cache.
**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local BGE cross-encoder).

**Inputs:** `state["doc_chunks"]`, `state["regulation_scope"]`

**Outputs written to state:** `retrieved_clauses`

**Logic per chunk:**
```python
for chunk in state["doc_chunks"]:
    for regulation in state["regulation_scope"]:
        col_size = vector_store.collection_size(regulation)
        results = retrieve_and_rerank(
            query=chunk["chunk_text"],
            namespace=regulation,
            top_k_candidates=min(10, col_size),
            top_k_final=min(5, col_size),
        )
    # Merge results across regulations, deduplicate by article_id
```

**`retrieve_and_rerank` function** (`backend/retrieval/vector_store.py`):
```python
def retrieve_and_rerank(query: str, namespace: str, top_k_candidates: int, top_k_final: int) -> list[dict]:
    embedding = embedder.embed(query)           # all-MiniLM-L6-v2, cached by SHA256(query)
    candidates = vector_store.query(
        namespace=namespace,
        query_embedding=embedding,
        n_results=top_k_candidates,
    )
    pairs = [(query, doc) for doc in candidates["documents"][0]]
    scores = reranker.predict(pairs)            # ms-marco cross-encoder
    ranked = sorted(zip(scores, candidates["documents"][0], candidates["metadatas"][0]), reverse=True)
    return [
        {"article_id": m["article_id"], "article_title": m["article_title"],
         "clause_text": d, "severity": m["severity"], "regulation": m["regulation"],
         "rerank_score": float(s)}
        for s, d, m in ranked[:top_k_final]
    ]
```

### 6.3 QwenRunner (`backend/debate/qwen_runner.py`)

**Singleton** — loaded once at process start, shared across all debate agents.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, re

class QwenRunner:
    # Override via QWEN_MODEL_ID env var — e.g., Qwen/Qwen2.5-0.5B-Instruct for dev/testing
    MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-8B")

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"           # GPU if available, else CPU
        )
        self.model.eval()

    def generate(self, prompt: str, thinking: bool = True, max_new_tokens: int = 1024) -> dict:
        """
        Returns:
            thinking_trace: str   — content of <think>...</think> (empty string if thinking=False)
            response: str         — text after </think> tag (or full output if no think tag)
            full_output: str      — complete raw model output
        """
        system_msg = (
            "You are an expert compliance auditor. Think step by step through the legal "
            "requirements before answering. Show your reasoning in <think>...</think> tags."
            if thinking else
            "You are an expert compliance auditor. Answer concisely and directly."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False     # deterministic — required for reproducibility
            )

        full_output = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        thinking_trace = ""
        response = full_output
        if "<think>" in full_output and "</think>" in full_output:
            thinking_trace = full_output.split("<think>")[1].split("</think>")[0].strip()
            response = full_output.split("</think>")[-1].strip()

        return {"thinking_trace": thinking_trace, "response": response, "full_output": full_output}

# Module-level singleton — import this, don't instantiate a new one
qwen = QwenRunner()
```

**Fallback for low-VRAM environments:**
```python
# If device has < 16GB VRAM, use 4-bit quantization
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
# Pass quantization_config=bnb_config to from_pretrained()
```

### 6.4 DebateProtocol (`backend/debate/protocol.py`)

Orchestrates a single `(chunk_text, clause)` debate round. Called by the LangGraph debate node for every pair.

```python
def run_debate(chunk_text: str, clause: dict, chunk_index: int, qwen: QwenRunner) -> DebateRecord:
    """
    Runs 3 sequential Qwen3 calls: Advocate → Challenger → Arbiter.
    Each call uses thinking=True. All thinking traces are captured.
    Returns a fully populated DebateRecord.
    """
    # ── Round 1: Advocate ────────────────────────────────────────────────
    adv_prompt = ADVOCATE_PROMPT.format(
        article_id=clause["article_id"], article_title=clause["article_title"],
        regulation=clause["regulation"].upper(), clause_text=clause["clause_text"],
        policy_chunk=chunk_text
    )
    adv_out = qwen.generate(adv_prompt, thinking=True)
    adv = safe_parse_json(adv_out["response"])  # fallback: {"argument": adv_out["response"], "cited_text": None, "confidence": 0.5}

    # ── Round 2: Challenger (receives Advocate's full output) ────────────
    chal_prompt = CHALLENGER_PROMPT.format(
        article_id=clause["article_id"], article_title=clause["article_title"],
        regulation=clause["regulation"].upper(), clause_text=clause["clause_text"],
        policy_chunk=chunk_text, advocate_full_output=adv_out["response"]
    )
    chal_out = qwen.generate(chal_prompt, thinking=True)
    chal = safe_parse_json(chal_out["response"])

    # ── Round 3: Arbiter (receives both) ─────────────────────────────────
    arb_prompt = ARBITER_PROMPT.format(
        article_id=clause["article_id"], article_title=clause["article_title"],
        regulation=clause["regulation"].upper(), clause_text=clause["clause_text"],
        policy_chunk=chunk_text,
        advocate_output=adv_out["response"], challenger_output=chal_out["response"]
    )
    arb_out = qwen.generate(arb_prompt, thinking=True)
    arb = safe_parse_json(arb_out["response"])

    # ── Hallucination Guard ───────────────────────────────────────────────
    cited = arb.get("cited_text")
    hallucination_flag = (
        arb.get("coverage", "Missing") in ("Full", "Partial")
        and (cited is None or cited not in chunk_text)
    )
    if hallucination_flag and cited is not None:
        arb["coverage"] = "Partial"     # downgrade when citation unverifiable

    return DebateRecord(
        article_id=clause["article_id"], article_title=clause["article_title"],
        regulation=clause["regulation"], chunk_index=chunk_index,
        advocate_argument=adv.get("argument", ""), advocate_cited_text=adv.get("cited_text"),
        advocate_confidence=adv.get("confidence", 0.5), advocate_thinking=adv_out["thinking_trace"],
        challenger_argument=chal.get("counterargument", ""), challenger_gap=chal.get("gap_identified", ""),
        challenger_confidence=chal.get("confidence", 0.5), challenger_thinking=chal_out["thinking_trace"],
        verdict=arb.get("coverage", "Missing"), risk_level=clause.get("severity", "Medium"),
        reasoning=arb.get("reasoning", ""), final_cited_text=cited,
        debate_summary=arb.get("debate_summary", ""), arbiter_thinking=arb_out["thinking_trace"],
        hallucination_flag=hallucination_flag
    )
```

**Deduplication rule:** When the same `article_id` appears in debate records from multiple chunks, take the record with the **best verdict** (`Full > Partial > Missing`) as the canonical per-article result.

### 6.5 ReporterAgent (`backend/agents/reporter.py`)

**Inputs:** `state["debate_records"]`, `state["doc_id"]`, `state["doc_type"]`, `state["regulation_scope"]`

**Outputs written to state:** `risk_score`, `risk_level`, `violation_report`, `poam`

**Risk scoring:**
```python
RISK_WEIGHTS     = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
COVERAGE_PENALTY = {"Missing": 1.0, "Partial": 0.5, "Full": 0.0}
RISK_THRESHOLDS  = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Critical": 4.0}

def compute_risk_score(canonical_records: list[DebateRecord]) -> tuple[float, str]:
    """canonical_records: one DebateRecord per article (best verdict already selected)"""
    if not canonical_records:
        return 0.0, "Low"
    total = sum(RISK_WEIGHTS[r["risk_level"]] * COVERAGE_PENALTY[r["verdict"]] for r in canonical_records)
    max_possible = sum(RISK_WEIGHTS[r["risk_level"]] for r in canonical_records)
    score = round(total / max_possible * 4, 2)
    level = next(k for k, v in sorted(RISK_THRESHOLDS.items(), key=lambda x: x[1]) if score <= v)
    return score, level
```

**Remediation generation:** One batched Qwen3-8B call for all violations. Prompt asks for 2-3 sentence remediation per violation with specific language to add.

**Hallucination rate** is stored as a fraction in `[0.0, 1.0]` (e.g., `0.3333` = 33.3%). Display layers should multiply by 100 for percentage display.

**POA&M report generation:**
```python
def generate_poam(violation_report: dict, debate_records: list, state: dict, doc_id: str) -> POAMReport:
    """Renders both Jinja2 templates → Markdown strings → PDF via pdf_renderer.markdown_to_pdf()"""
    # Files written to outputs/reports/{doc_id}/POA&M/
    assessment_path  = f"outputs/reports/{doc_id}/POA&M/assessment_report.pdf"
    remediation_path = f"outputs/reports/{doc_id}/POA&M/remediation_report.pdf"
    # assessment.py + remediation.py render Jinja2 → Markdown
    # pdf_renderer.markdown_to_pdf() converts Markdown → styled HTML → PDF (xhtml2pdf + reportlab)
    return POAMReport(assessment_report_path=assessment_path, remediation_report_path=remediation_path)
```

---

## 7. POA&M Report Specifications

The system generates two reports per document run, placed in `outputs/reports/{doc_id}/POA&M/`.

### 7.1 Assessment Report (`assessment_report.pdf`)

The Assessment Report is the comprehensive audit finding. It contains:

```
# Compliance Assessment Report
**Document:** {doc_id} — {doc_type}
**Regulation(s):** {regulation_scope joined}
**Assessment Date:** {generated_at}
**Overall Risk Score:** {risk_score} / 4.0 ({risk_level})
**Assessed By:** AI Compliance Monitoring Agent v1.0

---

## Executive Summary
{2-3 paragraph summary: what was assessed, overall finding, most critical gaps}

---

## Assessment Methodology
- Adversarial Debate Protocol (Advocate → Challenger → Arbiter)
- Model: Qwen/Qwen3-8B with explicit reasoning traces
- Regulatory Source: [regulation source URL] — Version: {regulation_version}
- Retrieval: RAG with BGE cross-encoder reranking

---

## Article-by-Article Findings

### {article_id} — {article_title} [{severity}] — Coverage: {verdict}

**Regulatory Requirement:**
{clause_text}

**Key Requirements Checklist:**
- [x] {requirement} ← checked if Full or Partial
- [ ] {requirement} ← unchecked if Missing

**Evidence Found:**
> "{final_cited_text}" ← exact quote from policy, or "No supporting language found."

**Debate Summary:**
{debate_summary}

**Finding:**
{reasoning}

**Hallucination Flag:** {hallucination_flag}

---
[repeat for each article evaluated]

---

## Risk Summary Table

| Article | Title | Coverage | Risk Level | SRS (if drift) |
|---|---|---|---|---|
| art_5 | Principles | Partial | Critical | — |
...

---

## Hallucination Analysis
Total evaluations: {n}
Flagged evaluations: {hallucination_flags}
Hallucination rate: {rate}%

---

## Appendix: Full Debate Transcripts

### Debate: {article_id} — Chunk {chunk_index}

**Advocate Thinking Trace:**
{advocate_thinking}

**Advocate Argument:**
{advocate_argument}

**Challenger Thinking Trace:**
{challenger_thinking}

**Challenger Argument:**
{challenger_argument}

**Arbiter Thinking Trace:**
{arbiter_thinking}

**Arbiter Verdict:**
{arbiter reasoning + verdict}
```

### 7.2 Remediation Report (`remediation_report.pdf`)

The Remediation Report is the actionable follow-up. It contains:

```
# Compliance Remediation Report
**Document:** {doc_id}
**Based On Assessment:** {assessment_date}
**Regulation(s):** {regulation_scope}
**Priority:** {risk_level}

---

## Remediation Summary
{n_violations} violations require remediation.
{n_critical} Critical · {n_high} High · {n_medium} Medium

Estimated remediation effort: {derived from severity counts}

---

## Remediation Actions

### ACTION-001 [CRITICAL] — {article_id}: {article_title}

**Current State:** {verdict} — {reasoning}

**Gap Identified:**
{challenger_gap}

**Required Action:**
Add the following language to the {section} section of the document:

> {specific remediation language — 2-3 sentences of exact text to add}

**Acceptance Criteria:**
The remediated document must satisfy:
{key_requirements that were missing, formatted as checklist}

**References:**
- {regulation} {article_id}: {source_url}
- {article_title} full text

---
[repeat ACTION-00N for each Missing or Partial finding, sorted by risk_level DESC]

---

## Re-evaluation Checklist

After applying remediations, re-run the pipeline on the updated document.
Expected outcome: risk_score < 1.0 (Low risk level).

---

## Drift Monitoring Note
{if drift_result is not None:}
This document was previously assessed on {v1_date}.
{regression_count} regressions detected since last assessment.
Regressions: {list of article_id + from_coverage + to_coverage + SRS}
{else:}
This is the initial assessment. Subsequent assessments will track compliance drift.
```

---

## 8. Data Contracts

All inter-component data structures. Do not deviate.

### RetrievedClause
```json
{
  "article_id": "art_17",
  "article_title": "Right to erasure ('right to be forgotten')",
  "clause_text": "...",
  "severity": "Critical",
  "regulation": "gdpr",
  "rerank_score": 0.847
}
```

### ViolationReport
```json
{
  "doc_id": "nc_001",
  "doc_type": "privacy_policy",
  "regulations": ["gdpr"],
  "risk_score": 2.8,
  "risk_level": "High",
  "articles_evaluated": 10,
  "violations": [
    {
      "article_id": "art_17",
      "article_title": "Right to erasure ('right to be forgotten')",
      "regulation": "gdpr",
      "verdict": "Missing",
      "risk_level": "Critical",
      "reasoning": "...",
      "final_cited_text": null,
      "debate_summary": "Advocate found no supporting text; Challenger identified complete absence; Arbiter concurred.",
      "remediation": "Add a clause stating: 'Users may request deletion of their personal data by contacting [DPO email]. We will process all valid erasure requests within 30 days, subject to legal retention obligations.'",
      "hallucination_flag": false
    }
  ],
  "hallucination_flags": 0,
  "hallucination_rate": 0.0,              // fraction in [0.0, 1.0] — NOT a percentage
  "generated_at": "2025-04-11T14:32:01Z",
  "model": "Qwen/Qwen3-8B",
  "regulation_versions": {"gdpr": "2016/679 — last checked 2025-04-01"}
}
```

### DriftResult
```json
{
  "doc_id": "comp_001",
  "v1_assessed_at": "2025-03-01T10:00:00Z",
  "v2_assessed_at": "2025-04-11T14:32:01Z",
  "risk_score_v1": 0.5,
  "risk_score_v2": 1.8,
  "risk_score_delta": 1.3,
  "regressions": [
    {
      "article_id": "art_17",
      "from_coverage": "Full",
      "to_coverage": "Missing",
      "risk_level": "Critical",
      "semantic_distance": 0.97,
      "semantic_regression_score": 7.76
    }
  ],
  "improvements": [],
  "regression_count": 1,
  "critical_regressions": ["art_17"],
  "max_srs": 7.76
}
```

### PipelineLogEntry
```json
{
  "run_id": "a3f1b2c4-...",
  "doc_id": "nc_001",
  "agent": "arbiter",
  "article_id": "art_17",
  "regulation": "gdpr",
  "chunk_index": 2,
  "timestamp": "2025-04-11T14:32:01.123Z",
  "input_hash": "sha256_first_16_chars",
  "raw_prompt": "You are the final compliance arbiter...[full prompt]",
  "thinking_trace": "Let me consider both arguments carefully...[full think content]",
  "raw_response": "{\"coverage\": \"Missing\", ...}",
  "structured_output": "{\"verdict\": \"Missing\", \"risk_level\": \"Critical\", ...}",
  "error": null
}
```

---

## 9. Prompt Templates

### Advocate Prompt
```
You are a compliance advocate reviewing an enterprise policy against a regulatory requirement.
Your role: find every possible reading of the policy that satisfies this requirement.
Be thorough and generous in interpretation — find supporting language wherever it exists.

{REGULATION} REQUIREMENT:
{article_id} — {article_title}
{clause_text}

POLICY SECTION UNDER REVIEW:
{policy_chunk}

Find the strongest possible argument that this policy satisfies the requirement.
If you find supporting language, quote it exactly.

Respond in this exact JSON format:
{{"argument": "Your argument for compliance here", "cited_text": "exact verbatim quote from policy or null if none found", "confidence": 0.0}}
```

### Challenger Prompt
```
You are a strict compliance auditor. A compliance advocate has argued:

ADVOCATE'S ARGUMENT:
{advocate_full_output}

Your role: challenge this argument. Find every gap, ambiguity, and omission that shows
this policy does NOT fully satisfy the regulatory requirement. You have seen the advocate's
best case — now find what they missed or overstated.

{REGULATION} REQUIREMENT:
{article_id} — {article_title}
{clause_text}

POLICY SECTION:
{policy_chunk}

Respond in this exact JSON format:
{{"counterargument": "Your challenge to the advocate's position", "gap_identified": "The specific element that is missing or insufficient", "confidence": 0.0}}
```

### Arbiter Prompt
```
You are the final compliance arbiter. You have heard both sides of a compliance debate.

ADVOCATE argued:
{advocate_output}

CHALLENGER argued:
{challenger_output}

{REGULATION} REQUIREMENT:
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

CRITICAL: cited_text must be copied verbatim from the policy section above. If coverage is Full or Partial, cited_text must not be null.
```

### Classifier Prompt
```
Classify this enterprise document and identify which compliance regulations apply.

Supported regulations: gdpr (EU personal data), hipaa (US health data / PHI), nist (security controls — NIST SP 800-53).
Rules:
- gdpr: EU personal data collection, processing, consent, data-subject rights, privacy policies.
- hipaa: US protected health information (PHI), covered entities, BAAs, breach notification.
- nist: Security SOPs, access control policies, incident response, system security plans.
- gdpr + nist can appear together (e.g., security SOP that also processes EU personal data).
- hipaa and gdpr are mutually exclusive (conflicting retention/consent rules).

Examples:
Document: "This Privacy Policy describes how Acme Corp collects and uses personal data of EU residents..."
→ {"doc_type": "privacy_policy", "regulation_scope": ["gdpr"], "confidence": 0.95, "reasoning": "EU personal data processing — GDPR applies."}

Document: "This Security SOP defines incident response and access control procedures for our systems..."
→ {"doc_type": "security_sop", "regulation_scope": ["nist"], "confidence": 0.88, "reasoning": "Security procedures align with NIST SP 800-53 controls (IR-4, AC-2, AU-2)."}

Document: "This Business Associate Agreement covers PHI handling between covered entities..."
→ {"doc_type": "vendor_agreement", "regulation_scope": ["hipaa"], "confidence": 0.92, "reasoning": "BAA covering PHI — HIPAA applies. GDPR excluded due to conflict."}

Document: "This Data Handling SOP governs EU employee data stored on our secure infrastructure..."
→ {"doc_type": "data_handling", "regulation_scope": ["gdpr", "nist"], "confidence": 0.84, "reasoning": "EU personal data (GDPR) stored on auditable secure systems (NIST)."}

Classify:
{doc_snippet}
Filename: {doc_path}

Output ONLY JSON:
{{"doc_type": "privacy_policy|security_sop|vendor_agreement|data_handling|breach_sop|other", "regulation_scope": ["gdpr"|"hipaa"|"nist"], "confidence": 0.0-1.0, "reasoning": "1-2 sentences"}}
```

### Remediation Prompt
```
You are a compliance consultant. A compliance audit has identified these violations.
For each violation, write the exact language that should be added to the policy document.

VIOLATIONS:
{violations_json}

For each violation, provide:
1. The exact clause text to add (2-3 sentences, ready to paste into a policy document)
2. Which section of the document it should appear in

Respond as JSON array:
[{{"article_id": "art_17", "section": "User Rights", "remediation_text": "Users may request the deletion of their personal data by contacting our Data Protection Officer at [dpo@company.com]. We will process all valid erasure requests within 30 days of receipt, subject to legal retention obligations under applicable law. To submit an erasure request, please provide your account details and specify the data you wish to have deleted."}}]
```

---

## 10. Regulation Update System

### 10.1 RegulationWatcher (`backend/regulation/watcher.py`)

Polls official regulatory sources for changes. Runs as a scheduled job (daily via `schedule` library or cron).

```python
REGULATION_SOURCES = {
    "gdpr": {
        "url": "https://gdpr-info.eu/",
        "parser": "gdpr_parser",
        "check_interval_hours": 24
    },
    "hipaa": {
        "url": "https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C",
        "parser": "hipaa_parser",
        "check_interval_hours": 168   # weekly — CFR updates are infrequent
    },
    "nist": {
        "url": "https://csrc.nist.gov/projects/cprt/catalog",
        "parser": "nist_parser",
        "check_interval_hours": 720   # monthly — SP 800-53 revisions are rare
    }
}

class RegulationWatcher:
    def check_for_updates(self, regulation: str) -> list[dict]:
        """
        Fetches current regulation text.
        Compares article-by-article against stored version using semantic similarity.
        Returns list of changed articles: [{article_id, old_content, new_content, similarity_score}]
        Articles with cosine_similarity < UPDATE_THRESHOLD (0.95) are flagged as changed.
        """

    def apply_updates(self, regulation: str, changed_articles: list[dict]) -> None:
        """
        For each changed article:
        1. Re-embed and upsert to Chroma (replaces old vector)
        2. Write entry to RegulationChangelog
        3. Flag all documents previously assessed against changed articles for re-evaluation
        """
```

### 10.2 RegulationChangelog (`backend/regulation/changelog.py`)

```sql
CREATE TABLE regulation_changelog (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    regulation       TEXT NOT NULL,
    article_id       TEXT NOT NULL,
    change_detected_at TEXT NOT NULL,
    old_content_hash TEXT,
    new_content_hash TEXT,
    similarity_score REAL,
    reindexed        INTEGER DEFAULT 0,    -- 1 when Chroma updated
    affected_doc_ids TEXT                  -- JSON array of doc_ids needing re-evaluation
);

CREATE TABLE document_regulation_versions (
    doc_id           TEXT NOT NULL,
    regulation       TEXT NOT NULL,
    article_id       TEXT NOT NULL,
    regulation_version_hash TEXT NOT NULL, -- hash of article content at time of evaluation
    evaluated_at     TEXT NOT NULL,
    needs_reevaluation INTEGER DEFAULT 0
);
```

**Re-evaluation flagging logic:**
When article `art_17` changes, query `document_regulation_versions` for all `doc_id` values where `article_id = "art_17"` and `regulation_version_hash != new_hash`. Set `needs_reevaluation = 1` for those rows. The pipeline checks this flag before running and warns the user if stale.

### 10.3 Model Improvement Strategy

**Approach: Retrieval-Augmented Fine-Tuning (RAFT) — not standard fine-tuning**

Standard fine-tuning on compliance text would require thousands of examples and risks catastrophic forgetting of the model's legal reasoning capability. Instead, use **RAFT** (Zhang et al., 2024):

1. **Dataset construction:** From existing debate records (pipeline logs), extract `(article_text, policy_chunk, debate_transcript, arbiter_verdict)` tuples where both annotators agreed on the label. These are high-quality training examples.

2. **RAFT training format:** Each example is: context = [regulation article + policy chunk + advocate argument + challenger argument], target = [arbiter reasoning + verdict]. The model learns to produce arbiter-quality reasoning given a debate context.

3. **When to fine-tune:** Once 500+ high-confidence debate records are accumulated (hallucination_flag=False, inter-model kappa ≥ 0.8 on that example). This is dataset-size gated.

4. **Fine-tuning target:** Fine-tune only the Arbiter role initially — it makes the final judgment and benefits most from domain adaptation. Advocate and Challenger can remain base Qwen3-8B.

5. **Infrastructure:** Use Unsloth + QLoRA for efficient fine-tuning on the 8B model. Training runs on a single A100.

```python
# Future: scripts/finetune_arbiter.py
# Dataset: outputs/logs/ — filter records where hallucination_flag=False and kappa >= 0.8
# Format: RAFT-style instruction tuning
# Library: unsloth + trl (SFTTrainer)
# Checkpoint: saved to models/arbiter_raft_v{n}/
```

---

## 11. LangGraph Graph (`backend/graph.py`)

```python
from langgraph.graph import StateGraph, END
from backend.agents.state import ComplianceState
from backend.agents.classifier import classifier_node
from backend.agents.retrieval_agent import retrieval_node
from backend.agents.debate_agent import debate_node     # calls DebateProtocol for each (chunk, clause) pair
from backend.agents.reporter import reporter_node
from backend.drift.detector import drift_node           # only runs if previous_report_path is set

def build_graph() -> StateGraph:
    graph = StateGraph(ComplianceState)

    graph.add_node("classifier", classifier_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("debate", debate_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("drift", drift_node)

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "retrieval")
    graph.add_edge("retrieval", "debate")
    graph.add_edge("debate", "reporter")

    # Drift node only runs if previous_report_path is provided
    graph.add_conditional_edges(
        "reporter",
        lambda state: "drift" if state.get("previous_report_path") else END,
        {"drift": "drift", END: END}
    )
    graph.add_edge("drift", END)

    return graph.compile()

compiled_graph = build_graph()

def run_pipeline(doc_path: str, previous_report_path: str | None = None) -> ComplianceState:
    from backend.ingestion.parser import DocumentParser
    from backend.ingestion.chunker import DocumentChunker
    from backend.logging.pipeline_log import flush_pipeline_log
    import hashlib, os

    doc_text = DocumentParser().parse(doc_path)
    doc_chunks = DocumentChunker().chunk(doc_text)
    doc_id = hashlib.sha256(doc_path.encode()).hexdigest()[:12]

    initial_state = ComplianceState(
        doc_id=doc_id, doc_path=doc_path, doc_text=doc_text, doc_chunks=doc_chunks,
        doc_type="", regulation_scope=[], classifier_confidence=0.0, classifier_reasoning="",
        retrieved_clauses=[], debate_records=[],
        risk_score=0.0, risk_level="Low", violation_report={}, poam={},
        previous_report_path=previous_report_path, drift_result=None,
        pipeline_log=[]
    )

    final_state = compiled_graph.invoke(initial_state)

    # Write outputs
    flush_pipeline_log(final_state["pipeline_log"], doc_id)
    report_path = f"outputs/reports/{doc_id}/raw/violation_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        import json; json.dump(final_state["violation_report"], f, indent=2)

    return final_state
```

---

## 12. Pipeline Log Store (`backend/logging/pipeline_log.py`)

```sql
-- SQLite schema — outputs/pipeline_logs.db
CREATE TABLE pipeline_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    doc_id            TEXT NOT NULL,
    agent             TEXT NOT NULL,        -- "classifier"|"retrieval"|"advocate"|"challenger"|"arbiter"|"reporter"|"drift"
    article_id        TEXT,                 -- null for non-debate agents
    regulation        TEXT,
    chunk_index       INTEGER,
    timestamp         TEXT NOT NULL,        -- ISO 8601
    input_hash        TEXT,                 -- SHA256[:16] of input
    raw_prompt        TEXT,
    thinking_trace    TEXT,                 -- Qwen3 <think> content (null for non-Qwen agents)
    raw_response      TEXT,
    structured_output TEXT,                 -- JSON string
    error             TEXT                  -- null if success
);
```

`make_log_entry()` helper:
```python
from datetime import datetime, timezone
from hashlib import sha256
import json

def make_log_entry(agent: str, input_data: dict | str, raw_prompt: str | None,
                   thinking_trace: str | None, raw_response: str | None,
                   structured_output: dict, article_id: str = None,
                   regulation: str = None, chunk_index: int = None) -> dict:
    return {
        "agent": agent, "article_id": article_id, "regulation": regulation,
        "chunk_index": chunk_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": sha256(str(input_data).encode()).hexdigest()[:16],
        "input_data": json.dumps(input_data) if isinstance(input_data, dict) else str(input_data),
        "raw_prompt": raw_prompt, "thinking_trace": thinking_trace,
        "raw_response": raw_response, "structured_output": json.dumps(structured_output)
    }
```

---

## 13. Drift Detector (`backend/drift/detector.py`)

```python
import numpy as np
from backend.retrieval.embedder import embedder

def cosine_distance(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

RISK_WEIGHTS  = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
COVERAGE_RANK = {"Full": 2, "Partial": 1, "Missing": 0}

def detect_drift(report_v1: dict, report_v2: dict) -> dict:
    v1 = {v["article_id"]: v for v in report_v1["violations"]}
    v2 = {v["article_id"]: v for v in report_v2["violations"]}
    regressions, improvements = [], []

    for art_id in set(v1) | set(v2):
        v1_cov = v1.get(art_id, {}).get("verdict", "Missing")
        v2_cov = v2.get(art_id, {}).get("verdict", "Missing")
        delta = COVERAGE_RANK[v1_cov] - COVERAGE_RANK[v2_cov]

        if delta > 0:   # regression
            c1 = v1.get(art_id, {}).get("final_cited_text") or ""
            c2 = v2.get(art_id, {}).get("final_cited_text") or ""
            if c1 and c2:
                sem_dist = cosine_distance(embedder.embed(c1), embedder.embed(c2))
            elif c1 and not c2:
                sem_dist = 1.0      # clause entirely removed
            else:
                sem_dist = 0.5
            risk_w = RISK_WEIGHTS.get(v1.get(art_id, {}).get("risk_level", "Medium"), 2)
            srs = round(delta * risk_w * (1 + sem_dist), 3)
            regressions.append({
                "article_id": art_id, "regulation": v1.get(art_id, {}).get("regulation", ""),
                "from_coverage": v1_cov, "to_coverage": v2_cov,
                "risk_level": v1.get(art_id, {}).get("risk_level", "Unknown"),
                "semantic_distance": round(sem_dist, 3),
                "semantic_regression_score": srs
            })
        elif delta < 0:   # improvement
            improvements.append({"article_id": art_id, "from": v1_cov, "to": v2_cov})

    return {
        "risk_score_v1": report_v1["risk_score"], "risk_score_v2": report_v2["risk_score"],
        "risk_score_delta": round(report_v2["risk_score"] - report_v1["risk_score"], 2),
        "regressions": sorted(regressions, key=lambda r: r["semantic_regression_score"], reverse=True),
        "improvements": improvements, "regression_count": len(regressions),
        "critical_regressions": [r for r in regressions if r["risk_level"] == "Critical"],
        "max_srs": max((r["semantic_regression_score"] for r in regressions), default=0.0)
    }
```

---

## 14. Evaluation Harness

The evaluation pipeline has two layers: a **scripted harness** that runs the full debate pipeline across `test_datasets/` and a **semantic-similarity judge** that scores results against ground truth without relying on the production debate verdict (so the judge stays diagnostic even when the underlying LLM regresses).

### 14.1 Workflow

```
test_datasets/{reg}/articles/*.pdf
            │
            ▼
   scripts/batch_evaluate.py        ← runs backend.graph.run_pipeline on every doc,
            │                          copies POA&M PDFs + violation_report.json
            │                          into outputs/POA&M/{doc_stem}/, writes metrics.json
            ▼
   scripts/compute_full_metrics.py  ← attaches RAGAS proxies, classification metrics,
            │                          and system-performance numbers per doc
            │                          → outputs/POA&M/_evaluation_report.json
            ▼
   scripts/enrich_metrics.py        ← (optional) recomputes retrieval-recall metrics
            │                          from outputs/pipeline_logs.db
            ▼
   scripts/evaluate_semantic.py     ← PRIMARY judge — sentence-transformer cosine + rule-based
            │                          classifier; produces per-doc semantic_metrics.json and
            │                          outputs/POA&M/_semantic_evaluation.json
            ▼
   scripts/generate_tables.py       ← renders the 6-table presentation report
                                       (Retrieval × Clause × Report) × (GDPR × HIPAA)
                                       → EVALUATION_TABLES.{md,pdf} + landscape PDF
```

Run all five scripts in order to regenerate every artifact under `outputs/POA&M/`. The semantic judge is independent of `batch_evaluate.py` and can be re-run alone whenever the ground-truth annotations change.

### 14.2 Semantic Judge (`scripts/evaluate_semantic.py`)

The semantic judge replaces the broken Qwen-0.5B baseline (which collapsed every verdict to `Missing`). It uses `sentence-transformers/all-MiniLM-L6-v2` cosine similarity between policy chunks and each focus article's `key_requirements`, then applies regulation-specific thresholds:

```python
THRESH = {
    "gdpr":  {"full_mean": 0.42, "full_cov": 0.55, "partial_mean": 0.33, "partial_cov": 0.25},
    "hipaa": {"full_mean": 0.38, "full_cov": 0.50, "partial_mean": 0.30, "partial_cov": 0.20},
}
```

A rule-based keyword classifier provides the doc-type / regulation routing fallback (this is what brings classifier accuracy from 50% → 100% versus the 0.5B model). The judge emits Full/Partial/Missing verdicts plus reformulated RAGAS proxies expressed as **hit-rate fractions** (not raw cosine means):

- `faithfulness` = fraction of assertive verdicts whose evidence cosine ≥ 0.30
- `answer_relevance` = fraction of articles where the mean top-similarity ≥ 0.30
- `context_precision` = fraction of articles where best-evidence cosine ≥ 0.40
- `hallucination_rate` = fraction of Full/Partial verdicts where evidence < 0.30

### 14.3 Experimental Conditions (legacy `run_evaluation.py`)

| ID | Description | LLM | RAG | Rerank | Debate |
|---|---|---|---|---|---|
| C1 | Naive baseline | gpt-4o-mini | No | No | No (single prompt) |
| C2 | RAG only | Qwen3-8B | Yes | No | No |
| C3 | RAG + rerank | Qwen3-8B | Yes | Yes | No |
| C4 | Full system | Qwen3-8B | Yes | Yes | Yes (3-agent debate) |
| C4-nothink | Ablation | Qwen3-8B (think OFF) | Yes | Yes | Yes |

**What each comparison isolates:**
- C2 vs C1 → RAG grounding value
- C3 vs C2 → cross-encoder reranking value
- C4 vs C3 → adversarial debate value
- C4 vs C4-nothink → Qwen3 thinking-trace value

The toggle for C4-nothink is the `--no-thinking` flag on `run_pipeline.py` (or `thinking=false` on `POST /api/v1/analyze`).

### 14.4 Metrics

Three classification perspectives are reported per regulation, since label imbalance (~77% Full, ~15% Missing, ~8% Partial in the n=60 article-level set) makes any single view misleading:

```python
def compute_metrics(predictions, ground_truth):
    """
    predictions: {doc_stem: {article_id: "Full"|"Partial"|"Missing"}}
    ground_truth: same shape, parsed from annotation PDFs.
    Returns three views:
      • Full-as-positive          (compliance certification view)
      • Non-compliant-as-positive (audit-finding view — operationally important)
      • Macro-averaged 3-class    (label-balanced)
    Plus exact-match accuracy and Cohen's Kappa.
    """
```

Additional metrics surfaced in `EVALUATION_TABLES.md`:

- **RAGAS Faithfulness** — target ≥ 0.85 (rubric value)
- **RAGAS Answer Relevance** — target ≥ 0.80
- **RAGAS Context Precision** — target ≥ 0.75
- **Hallucination rate** — `hallucination_flags / total_debate_evaluations` (also reported as a fraction in `[0.0, 1.0]` in `violation_report.json`)
- **Likert report-quality score** — 1-4: 4 = all artifacts present + 0 hallucinations; 3 = all + ≤10% hall; 2 = missing artifact or ≤25%; 1 = severe.
- **Debate consistency** (legacy) — % where Arbiter aligns with the side that matches ground truth.

---

## 15. Environment Setup

### Requirements (`requirements.txt`)
See the actual file at the repo root — the locked set includes `torch`, `transformers`, `accelerate`, `langgraph`, `langchain`, `chromadb`, `sentence-transformers`, `tiktoken`, `pymupdf`, `python-docx`, `jinja2`, `xhtml2pdf`, `markdown`, `fastapi`, `uvicorn`, `python-multipart`, `python-dotenv`, `scikit-learn`, `numpy`, `schedule`. No OpenAI / Anthropic API keys are required for production runs — every model is local.

### Environment Variables (`.env`)
```bash
# Model selection (Qwen3-8B in production, 0.5B-Instruct for dev iteration)
QWEN_MODEL_ID=Qwen/Qwen3-8B

# Optional — Hugging Face token to avoid download rate limits
HF_TOKEN=hf_...

# CORS allow-list for the frontend (comma-separated, or "*" for any origin)
AGENTIC_AUDIT_CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Regulation-watcher toggles
REGULATION_WATCH_ENABLED=true
UPDATE_THRESHOLD=0.95            # cosine similarity below this → article flagged as changed
```

### First-time Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # fill in QWEN_MODEL_ID + CORS origins as needed

# Build regulation indexes (run once per regulation; safe to re-run)
python scripts/index_regulations.py --regulation gdpr
python scripts/index_regulations.py --regulation hipaa
python scripts/index_regulations.py --regulation nist

# CLI: evaluate a single document end-to-end
python run_pipeline.py --doc test_datasets/gdpr/articles/gdpr_partial_known_quickdeals.pdf

# Same, with the C4-nothink ablation (no Qwen <think> traces)
python run_pipeline.py --doc <file> --no-thinking

# Drift detection: pass a previous violation_report.json
python run_pipeline.py --doc <file> --previous-report outputs/reports/<doc_id>/raw/violation_report.json

# API: start the FastAPI server (consumed by the frontend)
uvicorn backend.api.main:app --reload --port 8000

# Full evaluation pipeline (regenerates outputs/POA&M/)
python scripts/batch_evaluate.py
python scripts/compute_full_metrics.py
python scripts/enrich_metrics.py
python scripts/evaluate_semantic.py
python scripts/generate_tables.py
```

### API Cost Estimate

| Operation | Model | Estimated Cost |
|---|---|---|
| Classifier + Debate + Remediation | Qwen3-8B (local) | $0.00 |
| Embeddings (retrieval + drift + judge) | all-MiniLM-L6-v2 (local) | $0.00 |
| Reranker | ms-marco-MiniLM-L-6-v2 (local) | $0.00 |
| **Total per evaluation run** | | **$0.00** |

The system is fully self-contained. No outbound API calls happen at inference time.

---

## 16. Design Decisions

**Why adversarial debate instead of single-agent?**
Legal compliance is inherently adversarial — courts hear both sides before ruling. Single-agent evaluators anchor on superficial language matches. Debate forces the system to surface and resolve contradictions. Grounded in Du et al. 2023.

**Why Qwen3-8B locally?**
Three reasons: (1) thinking traces are visible, loggable, and a research artifact; (2) enables the think ON/OFF ablation impossible with black-box APIs; (3) zero marginal cost per evaluation run — critical for reproducibility.

**Why one Chroma vector per article (not chunked)?**
All 10 GDPR focus articles fit within 700 tokens. Chunking splits legal reasoning mid-paragraph. One article = one vector = unambiguous source ID on every retrieved result.

**Why SQLite?**
Zero infrastructure. ~3000 log rows for a full evaluation run. Fully portable. The log is also written as human-readable JSON to `outputs/logs/`.

**Why RAFT for model improvement instead of standard fine-tuning?**
RAFT preserves the model's general legal reasoning while teaching it domain-specific compliance patterns. Standard fine-tuning on a small dataset risks catastrophic forgetting. RAFT uses the pipeline's own output (debate transcripts) as training signal, creating a self-improving loop.

**Why exclude GDPR+HIPAA and HIPAA+SOC2?**
GDPR Art. 17 (erasure) directly contradicts HIPAA §164.530(j) (retention minimums for medical records — 6 years). Attempting joint evaluation would produce contradictory verdicts with no clear resolution. Cleaner to evaluate against each regulation independently and surface conflicts as a warning rather than a joint score.

---

## 17. Research Questions

| RQ | Question | Measured By |
|---|---|---|
| RQ1 | Does adversarial debate improve clause-level accuracy vs. single-agent? | F1: C4 vs C3 |
| RQ2 | Does RAG grounding reduce hallucinations vs. no-RAG? | Hallucination rate + RAGAS: C2/C3/C4 vs C1 |
| RQ3 | Does cross-encoder reranking improve retrieval precision? | F1 + RAGAS: C3 vs C2 |
| RQ4 | Do Qwen3 thinking traces improve compliance reasoning? | F1 + hallucination rate: C4 vs C4-nothink |

---

## 18. Known Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen3-8B VRAM OOM | High without GPU | Use `load_in_4bit=True` via bitsandbytes; test early on target hardware |
| Debate produces malformed JSON | Medium | `safe_parse_json()` with structured fallback on every Qwen call |
| Regulation source URL changes (watcher breaks) | Medium | Cache last-known-good HTML; alert on fetch failure; manual override path |
| Ground truth kappa < 0.5 | Low | Annotate most unambiguous articles first (art_6, art_17); resolve by team discussion |
| State schema change mid-build | Medium | Lock `state.py` on Day 1; any change requires updating this spec first |
| GDPR+HIPAA conflict surfaced by classifier | Low | Exclusion matrix enforced at classifier level; logs the conflict and reason |

---

## 19. Frontend Integration & API Contract

The backend exposes a versioned REST API at `/api/v1/*` consumed by the React/Next.js app under `frontend/`. CORS is permissive for the configured origins (`AGENTIC_AUDIT_CORS_ORIGINS`).

### 19.1 Server lifecycle

```bash
# Dev (auto-reload):
uvicorn backend.api.main:app --reload --port 8000

# Prod (single process — Qwen3-8B is the bottleneck, not the HTTP layer):
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The first `/analyze` request loads `Qwen3-8B` into memory (~16 GB FP16 / ~6 GB 4-bit). Subsequent requests reuse the singleton — keep the worker count at 1 unless you have multi-GPU hardware.

### 19.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness check + active regulation namespaces |
| `GET` | `/api/v1/regulations` | All active regulations with their focus articles |
| `POST` | `/api/v1/analyze` | Upload a document and run the full compliance pipeline |
| `GET` | `/api/v1/reports` | List every completed run (for a dashboard table) |
| `GET` | `/api/v1/reports/{doc_id}` | Full ViolationReport JSON for one run |
| `GET` | `/api/v1/reports/{doc_id}/assessment` | Download the Assessment PDF |
| `GET` | `/api/v1/reports/{doc_id}/remediation` | Download the Remediation PDF |

### 19.3 Request / response shapes

**`GET /api/v1/health`**
```json
{ "status": "ok", "active_regulations": ["gdpr", "hipaa", "nist"] }
```

**`GET /api/v1/regulations`**
```json
{
  "gdpr":  { "namespace": "gdpr",  "focus_articles": ["art_5", "art_6", "..."] },
  "hipaa": { "namespace": "hipaa", "focus_articles": ["hipaa_164_306", "..."] },
  "nist":  { "namespace": "nist",  "focus_articles": ["nist_ac-2", "..."] }
}
```

**`POST /api/v1/analyze`** — `multipart/form-data` with a single `file` field. Optional query param `thinking=false` for the C4-nothink ablation. Accepted MIME types: `.pdf`, `.docx`, `.txt`.

Response (synchronous; long-running — typical latency 6–8 minutes per document on Qwen3-8B):
```json
{
  "doc_id": "a3f1b2c4d5e6",
  "doc_filename": "company_privacy_policy.pdf",
  "doc_type": "privacy_policy",
  "regulations": ["gdpr"],
  "risk_score": 2.4,
  "risk_level": "Medium",
  "articles_evaluated": 10,
  "hallucination_rate": 0.0,
  "thinking_enabled": true,
  "assessment_report_url":  "/api/v1/reports/a3f1b2c4d5e6/assessment",
  "remediation_report_url": "/api/v1/reports/a3f1b2c4d5e6/remediation"
}
```

**`GET /api/v1/reports/{doc_id}`** — full `ViolationReport` (see Section 8). Use this to render the dashboard / per-article table.

**`GET /api/v1/reports/{doc_id}/{assessment|remediation}`** — `application/pdf` download. The filename is `{doc_stem}_{kind}_{doc_id}.pdf`.

**`GET /api/v1/reports`** — paginated list (currently unbounded — frontend should add client-side filters):
```json
{
  "reports": [
    {
      "doc_id": "a3f1b2c4d5e6",
      "doc_type": "privacy_policy",
      "regulations": ["gdpr"],
      "risk_score": 2.4,
      "risk_level": "Medium",
      "generated_at": "2026-04-19T14:32:01Z"
    }
  ]
}
```

### 19.4 Error model

| Status | Meaning |
|---|---|
| `400` | Unsupported file type, malformed multipart, missing `file` field |
| `404` | `doc_id` not found in `outputs/reports/` |
| `500` | Pipeline failure — check `outputs/pipeline_logs.db` for the run_id |

All error bodies follow FastAPI's default shape: `{"detail": "<message>"}`.

### 19.5 Suggested frontend flow

```
1. GET /api/v1/health              → confirm backend is up
2. GET /api/v1/regulations         → render the supported-regulations panel
3. POST /api/v1/analyze            → file upload (show "running ~7 min" banner)
4. GET  /api/v1/reports/{doc_id}   → render risk score, per-article table, hallucination flag
5. GET  /api/v1/reports/{doc_id}/{assessment,remediation}
                                    → download buttons (open in new tab)
6. GET  /api/v1/reports            → "Recent runs" sidebar
```

Long-running uploads are the only sharp edge. Recommended UX: show a determinate progress bar fed by the typical 6–8 min budget (the backend does not currently stream incremental progress — adding SSE/WebSocket is a future enhancement listed in Section 21).

### 19.6 Sample fetch

```ts
async function analyze(file: File, thinking = true) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`/api/v1/analyze?thinking=${thinking}`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
  return r.json();
}
```

---

## 20. Teammate Handoff Runbook

For the engineer picking up backend integration / building the frontend.

### 20.1 What's done
- Full LangGraph pipeline (Classifier → Retrieval → Debate → Reporter, with optional Drift) — `backend/graph.py`.
- FastAPI layer with all endpoints listed in Section 19 — `backend/api/`.
- Three regulation indexes (GDPR, HIPAA, NIST) loaded into `data/chroma_db/`.
- POA&M PDF generation via Jinja2 + xhtml2pdf — `backend/reports/`.
- Drift detection with Semantic Regression Score — `backend/drift/detector.py`.
- Adversarial debate protocol with cite-then-verify hallucination guard — `backend/debate/protocol.py`.
- Evaluation harness + 6-table presentation report — `scripts/evaluate_semantic.py`, `scripts/generate_tables.py`, `outputs/POA&M/EVALUATION_TABLES.pdf`.
- Pipeline logging to SQLite (every prompt, thinking trace, response) — `outputs/pipeline_logs.db`.

### 20.2 What's not yet done (good first issues)
- **Frontend** — `frontend/` is empty. Suggested stack: Next.js + Tailwind + shadcn/ui. The API contract in Section 19 is stable.
- **Streaming progress** — `/analyze` is synchronous. Add Server-Sent Events keyed by `run_id` so the UI can show debate-round-by-debate-round progress.
- **Auth** — none today. Add a token-based middleware before deploying beyond localhost.
- **SOC 2 / ISO 27001 indexing** — the `data/compliance/{soc2,iso27001}/` directories and `test_datasets/{soc2,gdpr_soc2_combined,hipaa_soc2_combined}/` are placeholders. Drop in `*_articles.json` and run `scripts/index_regulations.py --regulation <reg>`.
- **Multi-doc batching endpoint** — `POST /api/v1/analyze/batch` accepting multiple files would help auditors evaluate a doc set in one call.
- **Background workers** — Qwen3-8B is the bottleneck. Move `/analyze` onto a Celery / RQ worker queue once load justifies it.

### 20.3 Local bring-up (10 minutes on Apple Silicon / 5 minutes on a CUDA box)

```bash
git clone <repo> && cd Agentic_Audit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                         # set QWEN_MODEL_ID and AGENTIC_AUDIT_CORS_ORIGINS

# Smoke test the pipeline (uses cached models on second run)
python run_pipeline.py --doc test_datasets/gdpr/articles/gdpr_compliant_streamvibe.pdf

# Start the backend
uvicorn backend.api.main:app --reload --port 8000

# In another shell — verify the frontend's first three calls
curl -s http://localhost:8000/api/v1/health | jq
curl -s http://localhost:8000/api/v1/regulations | jq
curl -s -F "file=@test_datasets/gdpr/articles/gdpr_compliant_streamvibe.pdf" \
        http://localhost:8000/api/v1/analyze | jq
```

### 20.4 Code touch-points by task

| Frontend wants… | Backend file to edit |
|---|---|
| New regulation surfaced | `backend/agents/classifier.py::REGULATION_REGISTRY` + `data/compliance/{reg}/` + `scripts/index_regulations.py --regulation <reg>` |
| Different report layout | `backend/reports/templates/{assessment,remediation}.md.jinja` |
| Adjust risk-score weights | `backend/agents/reporter.py::RISK_WEIGHTS` |
| Different chunk size | `backend/ingestion/chunker.py` |
| Add a new endpoint | `backend/api/routes.py` (route) + plain function in the relevant `backend/<module>/` |
| New ablation flag | `run_pipeline.py` arg → thread through `ComplianceState["thinking_enabled"]`-style key in `backend/agents/state.py` |

### 20.5 Files the frontend never needs to touch

- Anything under `data/`, `outputs/`, `test_datasets/` — generated or evaluation-only.
- `scripts/*` — research-evaluation utilities, not part of production runtime.
- `run_evaluation.py` — superseded for results reporting by `scripts/evaluate_semantic.py` + `scripts/generate_tables.py`.

---

## 21. Future Enhancements (non-blocking)

- Server-Sent Events for incremental debate progress on `/analyze`.
- Authentication + per-user report scoping.
- Background worker queue (Celery / RQ) for long-running pipelines.
- SOC 2, ISO 27001, PCI-DSS indexing — the registry pattern in `backend/agents/classifier.py` is designed to absorb these.
- RAFT fine-tuning of the Arbiter once 500+ high-confidence debate records have accumulated (see Section 10.3).
- A ground-truth annotation UI replacing the current PDF-based annotator workflow.