"""Regulation Changelog — SQLite-backed audit trail of regulation changes
and document-to-regulation version mapping.

Two tables:

1. ``regulation_changelog`` — one row per detected change in any regulation
   article.  Tracks old/new content hashes, semantic similarity, and whether
   the change has been re-indexed into the vector store.

2. ``document_regulation_versions`` — records which regulation version
   (content hash) was in effect when a given document was last evaluated.
   When a regulation article changes, rows with a stale hash are flagged
   ``needs_reevaluation = 1`` so the scheduler knows which documents to
   re-run through the pipeline.

Usage:
    from backend.regulation.changelog import regulation_changelog
    regulation_changelog.log_change("gdpr", "art_17", old_hash, new_hash, 0.87)
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = str(_PROJECT_ROOT / "outputs" / "regulation_changelog.db")

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_CREATE_CHANGELOG_TABLE = """\
CREATE TABLE IF NOT EXISTS regulation_changelog (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    regulation          TEXT NOT NULL,
    article_id          TEXT NOT NULL,
    change_detected_at  TEXT NOT NULL,
    old_content_hash    TEXT,
    new_content_hash    TEXT,
    similarity_score    REAL,
    reindexed           INTEGER DEFAULT 0,
    affected_doc_ids    TEXT
);
"""

_CREATE_DOC_REG_VERSIONS_TABLE = """\
CREATE TABLE IF NOT EXISTS document_regulation_versions (
    doc_id                  TEXT NOT NULL,
    regulation              TEXT NOT NULL,
    article_id              TEXT NOT NULL,
    regulation_version_hash TEXT NOT NULL,
    evaluated_at            TEXT NOT NULL,
    needs_reevaluation      INTEGER DEFAULT 0,
    PRIMARY KEY (doc_id, regulation, article_id)
);
"""


class RegulationChangelog:
    """SQLite-backed changelog for regulation updates and document staleness tracking.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Parent directories are created
        automatically if they do not exist.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_tables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection with row-factory set to ``sqlite3.Row``."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Ensure both tables exist (idempotent)."""
        conn = self._connect()
        try:
            conn.execute(_CREATE_CHANGELOG_TABLE)
            conn.execute(_CREATE_DOC_REG_VERSIONS_TABLE)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Changelog operations
    # ------------------------------------------------------------------

    def log_change(
        self,
        regulation: str,
        article_id: str,
        old_hash: str | None,
        new_hash: str | None,
        similarity_score: float | None = None,
    ) -> int:
        """Record a detected change in a regulation article.

        Parameters
        ----------
        regulation : str
            Regulation namespace (e.g. ``"gdpr"``).
        article_id : str
            Identifier of the changed article (e.g. ``"art_17"``).
        old_hash : str | None
            SHA-256 hex digest of the previous article content, or *None* for
            a newly added article.
        new_hash : str | None
            SHA-256 hex digest of the new article content.
        similarity_score : float | None
            Cosine similarity between old and new article embeddings.

        Returns
        -------
        int
            The ``id`` (row id) of the newly inserted changelog entry.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            cursor = conn.execute(
                """\
                INSERT INTO regulation_changelog
                    (regulation, article_id, change_detected_at,
                     old_content_hash, new_content_hash, similarity_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (regulation, article_id, now, old_hash, new_hash, similarity_score),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        finally:
            conn.close()

    def mark_reindexed(self, changelog_id: int) -> None:
        """Mark a changelog entry as re-indexed in the vector store."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE regulation_changelog SET reindexed = 1 WHERE id = ?",
                (changelog_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def set_affected_doc_ids(self, changelog_id: int, doc_ids: list[str]) -> None:
        """Attach the list of affected document IDs to a changelog entry."""
        import json

        conn = self._connect()
        try:
            conn.execute(
                "UPDATE regulation_changelog SET affected_doc_ids = ? WHERE id = ?",
                (json.dumps(doc_ids), changelog_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Document-regulation version tracking
    # ------------------------------------------------------------------

    def flag_documents_for_reevaluation(
        self,
        regulation: str,
        article_id: str,
        new_hash: str,
    ) -> list[str]:
        """Flag every document whose stored regulation hash differs from *new_hash*.

        Sets ``needs_reevaluation = 1`` on matching rows and returns the list
        of affected ``doc_id`` values.

        Parameters
        ----------
        regulation : str
            Regulation namespace.
        article_id : str
            The article whose version has changed.
        new_hash : str
            SHA-256 hex digest of the updated article content.

        Returns
        -------
        list[str]
            Document IDs that now need re-evaluation.
        """
        conn = self._connect()
        try:
            # Identify stale documents.
            rows = conn.execute(
                """\
                SELECT doc_id FROM document_regulation_versions
                WHERE regulation = ? AND article_id = ?
                  AND regulation_version_hash != ?
                """,
                (regulation, article_id, new_hash),
            ).fetchall()
            affected_ids = [row["doc_id"] for row in rows]

            if affected_ids:
                conn.execute(
                    """\
                    UPDATE document_regulation_versions
                    SET needs_reevaluation = 1
                    WHERE regulation = ? AND article_id = ?
                      AND regulation_version_hash != ?
                    """,
                    (regulation, article_id, new_hash),
                )
                conn.commit()

            return affected_ids
        finally:
            conn.close()

    def record_evaluation(
        self,
        doc_id: str,
        regulation: str,
        article_id: str,
        content_hash: str,
    ) -> None:
        """Record (upsert) that *doc_id* has been evaluated against a specific
        regulation article version.

        On conflict the row is updated with the new hash and timestamp, and the
        ``needs_reevaluation`` flag is cleared.

        Parameters
        ----------
        doc_id : str
            The document that was evaluated.
        regulation : str
            Regulation namespace.
        article_id : str
            The regulation article.
        content_hash : str
            SHA-256 hex digest of the article content used during evaluation.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                """\
                INSERT INTO document_regulation_versions
                    (doc_id, regulation, article_id, regulation_version_hash,
                     evaluated_at, needs_reevaluation)
                VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT (doc_id, regulation, article_id)
                DO UPDATE SET
                    regulation_version_hash = excluded.regulation_version_hash,
                    evaluated_at            = excluded.evaluated_at,
                    needs_reevaluation      = 0
                """,
                (doc_id, regulation, article_id, content_hash, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_stale_documents(self, regulation: str | None = None) -> list[dict]:
        """Return documents that need re-evaluation.

        Parameters
        ----------
        regulation : str | None
            If provided, filter to a specific regulation namespace.  Otherwise
            return all stale documents across every regulation.

        Returns
        -------
        list[dict]
            Each dict contains ``doc_id``, ``regulation``, ``article_id``,
            ``regulation_version_hash``, and ``evaluated_at``.
        """
        conn = self._connect()
        try:
            if regulation:
                rows = conn.execute(
                    """\
                    SELECT doc_id, regulation, article_id,
                           regulation_version_hash, evaluated_at
                    FROM document_regulation_versions
                    WHERE needs_reevaluation = 1 AND regulation = ?
                    ORDER BY evaluated_at ASC
                    """,
                    (regulation,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """\
                    SELECT doc_id, regulation, article_id,
                           regulation_version_hash, evaluated_at
                    FROM document_regulation_versions
                    WHERE needs_reevaluation = 1
                    ORDER BY evaluated_at ASC
                    """,
                ).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_recent_changes(
        self, regulation: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Return the most recent changelog entries (newest first).

        Parameters
        ----------
        regulation : str | None
            Optional filter by regulation namespace.
        limit : int
            Maximum rows to return (default 50).

        Returns
        -------
        list[dict]
        """
        conn = self._connect()
        try:
            if regulation:
                rows = conn.execute(
                    """\
                    SELECT * FROM regulation_changelog
                    WHERE regulation = ?
                    ORDER BY change_detected_at DESC
                    LIMIT ?
                    """,
                    (regulation, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """\
                    SELECT * FROM regulation_changelog
                    ORDER BY change_detected_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
regulation_changelog = RegulationChangelog()
