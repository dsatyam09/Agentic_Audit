# Agentic AI Compliance Monitoring System

AI-powered compliance audit pipeline for **GDPR**, **HIPAA**, and **NIST SP 800-53**. Evaluates enterprise documents (privacy policies, SOPs, vendor agreements, breach response procedures) against regulatory standards and produces audit-ready POA&M reports.

## What makes it different

1. **Adversarial debate evaluation** — three Qwen3-8B agents (Advocate / Challenger / Arbiter) debate every clause instead of one model deciding alone.
2. **Visible thinking traces** — every Qwen `<think>` block is captured in `outputs/pipeline_logs.db` for full auditability.
3. **Adaptive regulation refresh** — `backend/regulation/watcher.py` polls EUR-Lex / eCFR / NIST CPRT and re-indexes only articles that semantically changed.
4. **Semantic Regression Score (SRS)** — drift detection ranks regressions between two versions of the same document.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # set QWEN_MODEL_ID + AGENTIC_AUDIT_CORS_ORIGINS

# CLI — run the full pipeline on one document
python run_pipeline.py --doc test_datasets/gdpr/articles/gdpr_compliant_streamvibe.pdf

# API — start the FastAPI server (consumed by the frontend)
uvicorn backend.api.main:app --reload --port 8000
```

## Repository map

| Path | What's there |
|---|---|
| `spec.md` | Single source of truth — schemas, contracts, runbooks |
| `architecture.md` | High-level walkthrough of the pipeline |
| `backend/` | All Python logic. API at `backend/api/`, LangGraph at `backend/graph.py` |
| `frontend/` | Reserved for the React/Next.js UI (see `spec.md` Section 19) |
| `data/compliance/` | Regulation source data (GDPR, HIPAA, NIST) |
| `data/chroma_db/` | Persistent vector store (auto-generated) |
| `test_datasets/` | Evaluation inputs + ground-truth annotation PDFs |
| `outputs/POA&M/` | Evaluation harness outputs + EVALUATION_TABLES.{md,pdf} |
| `scripts/` | Indexing, evaluation, and report-generation utilities |

## API endpoints

All under `/api/v1/`. See `spec.md` Section 19 for full request/response shapes.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + active regulations |
| `GET` | `/regulations` | Active namespaces and focus articles |
| `POST` | `/analyze` | Upload PDF/DOCX/TXT, run pipeline, return `doc_id` |
| `GET` | `/reports` | List completed runs |
| `GET` | `/reports/{doc_id}` | Full ViolationReport JSON |
| `GET` | `/reports/{doc_id}/{assessment\|remediation}` | Download PDF report |

## Documentation

- **Engineering spec:** [`spec.md`](./spec.md) — every schema, prompt, and contract.
- **Architecture walkthrough:** [`architecture.md`](./architecture.md) — pipeline flow with diagrams.
- **Teammate handoff runbook:** `spec.md` Section 20.
