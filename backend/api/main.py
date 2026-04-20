"""FastAPI application entry point.

Start the server:
    uvicorn backend.api.main:app --reload --port 8000

The API is designed for the frontend team.  All endpoints are under /api/v1/.

CORS allow-list is read from the ``AGENTIC_AUDIT_CORS_ORIGINS`` environment
variable (comma-separated).  The default allows only the local dev frontend.
Set ``AGENTIC_AUDIT_CORS_ORIGINS="*"`` to explicitly open up for testing.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "AGENTIC_AUDIT_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    ).strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Agentic Audit API",
    description="AI-powered compliance monitoring — GDPR, HIPAA, NIST SP 800-53",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(router, prefix="/api/v1")
