"""FastAPI application entry point.

Start the server:
    uvicorn backend.api.main:app --reload --port 8000

The API is under /api/v1/. When ``frontend/dist`` exists (``npm run build``),
the root URL serves the Web UI and ``/assets`` serves Vite bundles.
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"

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

if _FRONTEND_DIST.is_dir() and (_FRONTEND_DIST / "index.html").is_file():
    _assets = _FRONTEND_DIST / "assets"
    if _assets.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=_assets),
            name="ui-assets",
        )

    @app.get("/")
    def serve_ui():
        return FileResponse(_FRONTEND_DIST / "index.html")
else:

    @app.get("/")
    def root_placeholder():
        return {
            "service": "Agentic Audit API",
            "docs": "/docs",
            "ui": "Build the Web UI: cd frontend && npm install && npm run build",
        }
