"""
Pipeline Log Store
==================
SQLite-backed logger that captures every agent invocation in the compliance
pipeline: prompts, thinking traces, raw responses, and structured outputs.

Database:  outputs/pipeline_logs.db
JSON logs: outputs/logs/{doc_id}_{run_id}.json
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "outputs" / "pipeline_logs.db"
_LOGS_DIR = _PROJECT_ROOT / "outputs" / "logs"

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    doc_id            TEXT NOT NULL,
    agent             TEXT NOT NULL,
    article_id        TEXT,
    regulation        TEXT,
    chunk_index       INTEGER,
    timestamp         TEXT NOT NULL,
    input_hash        TEXT,
    raw_prompt        TEXT,
    thinking_trace    TEXT,
    raw_response      TEXT,
    structured_output TEXT,
    error             TEXT
);
"""

_INSERT_SQL = """\
INSERT INTO pipeline_logs (
    run_id, doc_id, agent, article_id, regulation, chunk_index,
    timestamp, input_hash, raw_prompt, thinking_trace,
    raw_response, structured_output, error
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def make_log_entry(
    agent: str,
    input_data: dict | str,
    raw_prompt: str | None,
    thinking_trace: str | None,
    raw_response: str | None,
    structured_output: dict,
    article_id: str | None = None,
    regulation: str | None = None,
    chunk_index: int | None = None,
) -> dict:
    """Build a single log-entry dict ready to be appended to the pipeline log.

    Parameters
    ----------
    agent : str
        Name of the pipeline agent (e.g. "classifier", "advocate").
    input_data : dict | str
        The data fed into the agent. Used to compute a reproducible hash.
    raw_prompt : str | None
        The full prompt string sent to the LLM (None for non-LLM steps).
    thinking_trace : str | None
        Content extracted from ``<think>`` blocks (Qwen3). None otherwise.
    raw_response : str | None
        The complete LLM response before any parsing.
    structured_output : dict
        Parsed/validated output of the agent step.
    article_id : str | None
        Regulation article being evaluated (None for non-debate agents).
    regulation : str | None
        Regulation framework identifier (e.g. "hipaa", "gdpr").
    chunk_index : int | None
        Index of the document chunk being processed.

    Returns
    -------
    dict
        A log entry with an ISO-8601 UTC timestamp and SHA-256[:16] input hash.
    """
    input_str = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
    return {
        "agent": agent,
        "article_id": article_id,
        "regulation": regulation,
        "chunk_index": chunk_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": sha256(str(input_data).encode()).hexdigest()[:16],
        "input_data": input_str,
        "raw_prompt": raw_prompt,
        "thinking_trace": thinking_trace,
        "raw_response": raw_response,
        "structured_output": json.dumps(structured_output),
    }


def flush_pipeline_log(
    log_entries: list[dict],
    doc_id: str,
    run_id: str | None = None,
) -> None:
    """Persist all accumulated log entries to SQLite and a JSON sidecar.

    Parameters
    ----------
    log_entries : list[dict]
        Entries produced by :func:`make_log_entry`.
    doc_id : str
        Document identifier (typically a truncated SHA-256 of the file path).
    run_id : str | None
        Pipeline run identifier. Auto-generated as a UUID-4 when *None*.
    """
    if not log_entries:
        return

    if run_id is None:
        run_id = uuid.uuid4().hex

    # ------------------------------------------------------------------
    # 1. Write to SQLite
    # ------------------------------------------------------------------
    os.makedirs(_DB_PATH.parent, exist_ok=True)

    conn = sqlite3.connect(str(_DB_PATH))
    try:
        conn.execute(_CREATE_TABLE_SQL)
        rows = []
        for entry in log_entries:
            rows.append((
                run_id,
                doc_id,
                entry.get("agent", ""),
                entry.get("article_id"),
                entry.get("regulation"),
                entry.get("chunk_index"),
                entry.get("timestamp", datetime.now(timezone.utc).isoformat()),
                entry.get("input_hash"),
                entry.get("raw_prompt"),
                entry.get("thinking_trace"),
                entry.get("raw_response"),
                entry.get("structured_output"),
                entry.get("error"),
            ))
        conn.executemany(_INSERT_SQL, rows)
        conn.commit()
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # 2. Write human-readable JSON sidecar
    # ------------------------------------------------------------------
    os.makedirs(_LOGS_DIR, exist_ok=True)
    json_path = _LOGS_DIR / f"{doc_id}_{run_id}.json"

    sidecar = {
        "run_id": run_id,
        "doc_id": doc_id,
        "entry_count": len(log_entries),
        "entries": log_entries,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, ensure_ascii=False)
