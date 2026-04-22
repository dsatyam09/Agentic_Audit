# Agentic Audit - System Architecture

## What the system does

Given an enterprise document (PDF, DOCX, or TXT), the system:
1. Identifies what kind of compliance document it is and which regulations apply
2. Retrieves the most relevant regulation articles for each section of the document
3. Runs three AI agents in an adversarial debate to decide whether each article is satisfied
4. Scores the overall compliance risk and generates remediation language
5. Optionally tracks how compliance has changed since a previous audit (drift detection)
6. Produces two PDF reports: a full Assessment Report and an actionable Remediation Report

Everything runs locally - no OpenAI API, no external calls, zero cost per run.

---

## How a PDF gets evaluated - answering the key question

**Does submitting a PDF check against all three regulations at once?**

No. The pipeline routes each document to the regulation(s) that actually apply to it:

```
PDF submitted
     │
     ▼
Classifier reads the first ~1500 chars + filename
     │
     ├─ privacy_policy          → ["gdpr"]
     ├─ security_sop            → ["nist"]
     ├─ vendor_agreement (PHI)  → ["hipaa"]
     ├─ breach_sop              → ["hipaa"]
     ├─ data_handling (EU)      → ["gdpr", "nist"]   ← can be multi-reg
     └─ other                   → ["gdpr"]  (safe default)
     │
     ▼
Retrieval + Debate only runs against the assigned regulation(s)
     │
     ▼
POA&M reports reference only the relevant regulatory framework
```

**HIPAA + GDPR are mutually exclusive** - they have conflicting data-retention and
consent rules. If the classifier assigns both, the system keeps only the one that
best matches the document type. NIST can coexist with either GDPR or HIPAA.

---

## Active regulations

| Regulation | Namespace | Articles indexed | Focus articles |
|---|---|---|---|
| GDPR (EU) | `gdpr` | 10 | Art. 5, 6, 7, 13, 14, 17, 25, 32, 33, 44 |
| HIPAA (US Health) | `hipaa` | 144 | §164.306/308/310/312/316/404/408/502/524/530 |
| NIST SP 800-53 | `nist` | 324 | AC-2, AC-3, AU-2, IA-2, IR-4, RA-3, SC-8, SI-2, CM-6, CP-9 |

---

## Repository layout

```
Agentic_Audit/
│
├── backend/                    ← all Python logic
│   ├── agents/
│   │   ├── state.py            ← ComplianceState TypedDict (DO NOT CHANGE without spec update)
│   │   ├── classifier.py       ← doc_type + regulation routing
│   │   ├── retrieval_agent.py  ← RAG node
│   │   ├── debate_agent.py     ← orchestrates all (chunk, article) debates
│   │   └── reporter.py         ← risk scoring, remediation, PDF reports
│   ├── debate/
│   │   ├── protocol.py         ← Advocate / Challenger / Arbiter prompts
│   │   └── qwen_runner.py      ← Qwen singleton (env-configurable model ID)
│   ├── ingestion/
│   │   ├── parser.py           ← PDF/DOCX/TXT → plain text
│   │   └── chunker.py          ← 512-token overlapping chunks
│   ├── retrieval/
│   │   ├── embedder.py         ← all-MiniLM-L6-v2 with SHA-256 cache
│   │   ├── reranker.py         ← ms-marco cross-encoder
│   │   └── vector_store.py     ← ChromaDB wrapper, one collection per regulation
│   ├── drift/
│   │   └── detector.py         ← SRS computation + drift_node
│   ├── regulation/
│   │   ├── watcher.py          ← polls sources for regulation updates
│   │   └── differ.py           ← semantic diff between article versions
│   ├── reports/
│   │   ├── assessment.py       ← renders assessment_report via Jinja2
│   │   ├── remediation.py      ← renders remediation_report via Jinja2
│   │   ├── pdf_renderer.py     ← Markdown → HTML → PDF (xhtml2pdf)
│   │   └── templates/
│   │       ├── assessment.md.jinja
│   │       └── remediation.md.jinja
│   ├── logging/
│   │   └── pipeline_log.py     ← SQLite + JSON sidecar per run
│   ├── api/                    ← FastAPI layer for the frontend
│   │   ├── main.py             ← FastAPI app + CORS
│   │   └── routes.py           ← all API endpoints (/analyze, /reports, /health)
│   └── graph.py                ← LangGraph StateGraph wiring
│
├── data/
│   ├── compliance/             ← regulation source data (ground truth)
│   │   ├── gdpr/
│   │   │   ├── gdpr_raw.json       ← 272 raw records
│   │   │   └── gdpr_articles.json  ← 10 curated focus articles
│   │   ├── hipaa/
│   │   │   └── hipaa_articles.json ← 144 CFR sections (§160/164)
│   │   └── nist/
│   │       └── nist_articles.json  ← 324 SP 800-53 controls
│   └── chroma_db/              ← persistent ChromaDB (auto-generated, gitignored)
│
├── outputs/                    ← all pipeline outputs (gitignored)
│   ├── reports/{doc_id}/
│   │   ├── POA&M/
│   │   │   ├── assessment_report.pdf
│   │   │   └── remediation_report.pdf
│   │   └── raw/
│   │       └── violation_report.json
│   ├── drift/{doc_id}_drift_{ts}.json
│   ├── pipeline_logs.db        ← SQLite (all runs)
│   └── logs/{doc_id}_{run_id}.json
│
├── scripts/
│   ├── index_regulations.py   ← embed articles → ChromaDB (run once per regulation)
│   └── prepare_dataset.py     ← raw JSON → enriched articles.json
│
├── frontend/                  ← React/Next.js app (future phase)
│   └── (placeholder)
│
├── run_pipeline.py            ← CLI: python run_pipeline.py --doc <file>
├── run_evaluation.py          ← CLI: evaluation harness (5 conditions)
├── requirements.txt
├── .env                       ← QWEN_MODEL_ID, REGULATION_WATCH_ENABLED
└── architecture.md            ← this file
```

---

## High-level pipeline

```
Input Document (PDF / DOCX / TXT)
         │
         ▼
   ┌─────────────┐
   │  INGESTION  │  parse text → split into 512-token chunks (50-token overlap)
   └──────┬──────┘
          │  doc_text, doc_chunks
          ▼
   ┌─────────────┐
   │  CLASSIFIER │  Qwen LLM: doc_type + regulation_scope + exclusion matrix
   └──────┬──────┘
          │  e.g. doc_type="security_sop", regulation_scope=["nist"]
          ▼
   ┌─────────────┐
   │  RETRIEVAL  │  for each chunk:
   │             │    embed → ChromaDB ANN (top-10) → cross-encoder rerank (top-5)
   └──────┬──────┘
          │  retrieved_clauses: top-5 regulation articles per chunk
          ▼
   ┌─────────────┐
   │   DEBATE    │  for each (chunk, article) pair - 3 sequential Qwen calls:
   │             │    Advocate → Challenger → Arbiter + hallucination guard
   └──────┬──────┘
          │  debate_records: verdict + risk_level + thinking traces per article
          ▼
   ┌─────────────┐
   │  REPORTER   │  deduplicate per article, score risk, generate remediation,
   │             │  render Jinja2 templates → PDFs
   └──────┬──────┘
          │  (only if previous_report_path provided)
          ▼
   ┌─────────────┐
   │    DRIFT    │  compare v1 vs v2 → Semantic Regression Score per article
   └──────┬──────┘
          ▼
       OUTPUTS
  ├─ assessment_report.pdf
  ├─ remediation_report.pdf
  ├─ violation_report.json
  └─ pipeline_logs.db + {run_id}.json
```

---

## Shared state

All nodes read from and write to a single `ComplianceState` TypedDict
(`backend/agents/state.py`). LangGraph merges `pipeline_log` automatically.

```
ComplianceState
├── doc_id, doc_path, doc_text, doc_chunks          ← ingestion
├── doc_type, regulation_scope, classifier_*        ← classifier output
├── retrieved_clauses                               ← retrieval output
├── debate_records                                  ← debate output
├── risk_score, risk_level, violation_report, poam  ← reporter output
├── previous_report_path, drift_result              ← drift (optional)
└── pipeline_log (Annotated[list, operator.add])    ← auto-merged
```

---

## Step-by-step detail

### Step 1 - Ingestion (before the graph)

```
DocumentParser.parse(doc_path)
├── .pdf  → PyMuPDF  → page.get_text()
├── .docx → python-docx → paragraph text
└── .txt  → plain read

DocumentChunker.chunk(doc_text)
├── tiktoken cl100k_base encoding
├── 512-token window, 50-token overlap
└── returns [{chunk_index, chunk_text, char_start, char_end}]
```

### Step 2 - Classifier

```
doc_text[:1500] + filename
    → Qwen (thinking=False, max_new_tokens=256)
    → JSON: {doc_type, regulation_scope, confidence, reasoning}
    → filter to active regulations
    → enforce exclusion matrix (HIPAA ∩ GDPR = ∅)
```

**Regulation routing summary:**

| doc_type | Default regulation(s) |
|---|---|
| privacy_policy | gdpr |
| data_handling | gdpr |
| security_sop | nist |
| vendor_agreement | hipaa |
| breach_sop | hipaa |
| other | gdpr |

### Step 3 - Retrieval

```
For each chunk × each regulation in scope:
    1. embed(chunk_text)  →  all-MiniLM-L6-v2 (384-dim, SHA-256 cached)
    2. ChromaDB ANN query  →  top-10 candidates (cosine similarity)
    3. Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)  →  top-5
    4. Deduplicate by article_id across regulations
```

Two-stage retrieval: ANN is fast but imprecise; cross-encoder reads both texts
together for much higher relevance precision.

### Step 4 - Debate (core of the system)

For every `(chunk_text, article)` pair - 3 sequential Qwen calls:

```
ADVOCATE   - find every way the policy satisfies the requirement
    ↓           (thinking=True, <think> trace captured)
CHALLENGER - find every gap, ambiguity, omission in the advocate's argument
    ↓           (sees full advocate output, thinking=True)
ARBITER    - weigh both sides, issue final verdict
    ↓           (thinking=True)
HALLUCINATION GUARD
    └─ if verdict ∈ {Full, Partial} and cited_text not verbatim in chunk → flag + downgrade
```

**Verdict options:** Full / Partial / Missing
**Risk levels:** Critical / High / Medium / Low

### Step 5 - Reporter

```
1. Deduplicate: per article_id keep the BEST verdict (Full > Partial > Missing)
2. Risk score:  sum(RISK_WEIGHTS[risk_level] × COVERAGE_PENALTY[verdict]) / max × 4.0
                RISK_WEIGHTS={Critical:4, High:3, Medium:2, Low:1}
                COVERAGE_PENALTY={Missing:1.0, Partial:0.5, Full:0.0}
3. Remediation: Qwen generates exact policy language to add for each violation
4. PDF reports: Jinja2 → Markdown → xhtml2pdf
5. violation_report.json written to outputs/reports/{doc_id}/raw/
```

### Step 6 - Drift (optional)

```
SRS = coverage_rank_drop × risk_weight × (1 + cosine_distance(cited_v1, cited_v2))

~4.0 = clause fully removed (Full→Missing, Critical article)
~2.0 = clause weakened (Full→Partial, High article)
~1.5 = language subtly changed but verdict held
```

---

## API for the frontend

Server start: `uvicorn backend.api.main:app --reload --port 8000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Service status + active regulations |
| GET | `/api/v1/regulations` | List namespaces + focus articles |
| POST | `/api/v1/analyze` | Upload file → run pipeline → returns `doc_id` |
| GET | `/api/v1/reports` | List all completed runs |
| GET | `/api/v1/reports/{doc_id}` | Full ViolationReport JSON |
| GET | `/api/v1/reports/{doc_id}/assessment` | Download Assessment PDF |
| GET | `/api/v1/reports/{doc_id}/remediation` | Download Remediation PDF |

**Typical frontend flow:**
```
1. POST /api/v1/analyze  (multipart file)
   ← {doc_id, risk_score, risk_level, assessment_report_url, remediation_report_url}

2. GET /api/v1/reports/{doc_id}
   ← full ViolationReport JSON (for a dashboard table)

3. GET /api/v1/reports/{doc_id}/assessment
   ← PDF download
```

---

## Models (all local, no API key required)

| Model | Role | Size |
|---|---|---|
| `Qwen/Qwen3-8B` (production) | Classifier, Advocate, Challenger, Arbiter, Remediation | ~16 GB |
| `Qwen/Qwen2.5-0.5B-Instruct` (dev) | Same roles, fast iteration | ~1 GB |
| `all-MiniLM-L6-v2` | Embeddings (retrieval + drift cosine distance) | ~90 MB |
| `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker | ~70 MB |

Switch via `QWEN_MODEL_ID` in `.env`.

---

## LangGraph wiring

```python
graph = StateGraph(ComplianceState)
graph.add_node("classifier", classifier_node)
graph.add_node("retrieval",  retrieval_node)
graph.add_node("debate",     debate_node)
graph.add_node("reporter",   reporter_node)
graph.add_node("drift",      drift_node)

graph.set_entry_point("classifier")
graph.add_edge("classifier", "retrieval")
graph.add_edge("retrieval",  "debate")
graph.add_edge("debate",     "reporter")
graph.add_conditional_edges(        # drift only runs when a baseline exists
    "reporter",
    lambda state: "drift" if state.get("previous_report_path") else END,
    {"drift": "drift", END: END},
)
graph.add_edge("drift", END)
```
