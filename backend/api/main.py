"""FastAPI application entry point.

Start the server:
    uvicorn backend.api.main:app --reload --port 8000

The API is designed for the frontend team.  All endpoints are under /api/v1/.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(
    title="Agentic Audit API",
    description="AI-powered compliance monitoring — GDPR, HIPAA, NIST SP 800-53",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
