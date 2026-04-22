"""Regulation Watcher — monitors official regulation sources for updates,
computes semantic diffs, and triggers re-indexing and document re-evaluation.

The watcher is designed to run on a configurable schedule (see
``REGULATION_SOURCES``).  When called, ``check_for_updates`` compares the
current live articles against the embeddings already stored in ChromaDB.
Articles whose cosine similarity drops below ``UPDATE_THRESHOLD`` (env var,
default 0.95) are flagged as changed.  ``apply_updates`` then re-embeds the
new text, upserts to ChromaDB, logs the change, and flags stale documents.

Usage:
    from backend.regulation.watcher import regulation_watcher

    changed = regulation_watcher.check_for_updates("gdpr")
    if changed:
        regulation_watcher.apply_updates("gdpr", changed)
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from backend.regulation.changelog import regulation_changelog
from backend.regulation.differ import semantic_diff
from backend.retrieval.embedder import embedder
from backend.retrieval.vector_store import vector_store

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPDATE_THRESHOLD: float = float(os.getenv("UPDATE_THRESHOLD", "0.95"))

REGULATION_SOURCES: dict[str, dict[str, Any]] = {
    "gdpr": {
        "url": "https://gdpr-info.eu/",
        "check_interval_hours": 24,
    },
    "soc2": {
        "url": "https://www.aicpa.org/topic/audit-assurance/audit-and-assurance-greater-than-soc-2",
        "check_interval_hours": 168,
    },
    "hipaa": {
        "url": "https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html",
        "check_interval_hours": 168,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fetch_current_articles(regulation: str) -> list[dict[str, str]]:
    """Fetch the latest articles for *regulation* from the vector store.

    Returns a list of dicts with keys ``article_id``, ``content``, and
    ``content_hash``.  In a production deployment this would scrape or call
    an API endpoint for the official source; here it reads whatever is
    already indexed in ChromaDB as the "current" baseline and expects the
    caller to supply new content via the ``changed_articles`` list passed
    to :meth:`apply_updates`.

    .. note::
        A real scraper / API adapter per regulation should be plugged in
        here.  For now the vector-store contents act as the canonical
        "currently known" version.
    """
    collection = vector_store.get_or_create_collection(regulation)
    if collection.count() == 0:
        return []

    # Retrieve all stored articles.  ChromaDB ``get()`` with no filter
    # returns everything in the collection.
    result = collection.get(include=["documents", "metadatas"])

    articles: list[dict[str, str]] = []
    for doc_id, doc_text, meta in zip(
        result["ids"], result["documents"], result["metadatas"]  # type: ignore[arg-type]
    ):
        articles.append({
            "article_id": meta.get("article_id", doc_id),
            "content": doc_text,
            "content_hash": _content_hash(doc_text),
        })
    return articles


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegulationWatcher:
    """Monitors regulation sources for semantic changes and orchestrates
    re-indexing and document re-evaluation.
    """

    def __init__(self, threshold: float | None = None) -> None:
        self._threshold = threshold if threshold is not None else UPDATE_THRESHOLD

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def check_for_updates(
        self,
        regulation: str,
        new_articles: list[dict[str, str]] | None = None,
    ) -> list[dict]:
        """Compare current vs stored articles and return those that changed.

        Parameters
        ----------
        regulation : str
            Regulation namespace (e.g. ``"gdpr"``).
        new_articles : list[dict] | None
            The fresh article data to compare against what is stored.  Each
            dict must contain ``article_id`` and ``content``.  When *None*
            the method falls back to ``_fetch_current_articles`` (which in a
            production system would scrape the official source).

        Returns
        -------
        list[dict]
            Changed articles, each with keys: ``article_id``, ``old_content``,
            ``new_content``, ``old_hash``, ``new_hash``, ``cosine_similarity``.
        """
        if regulation not in REGULATION_SOURCES:
            logger.warning("Unknown regulation '%s' — skipping update check.", regulation)
            return []

        # Retrieve what we currently have stored.
        stored = _fetch_current_articles(regulation)
        stored_by_id: dict[str, dict[str, str]] = {
            a["article_id"]: a for a in stored
        }

        # If no explicit new articles are provided, we have nothing to
        # compare against (the real scraper would supply these).
        if new_articles is None:
            logger.info(
                "No new article payload supplied for '%s'; nothing to compare.",
                regulation,
            )
            return []

        changed: list[dict] = []

        for article in new_articles:
            art_id = article["article_id"]
            new_content = article["content"]
            new_hash = _content_hash(new_content)

            stored_article = stored_by_id.get(art_id)

            if stored_article is None:
                # Entirely new article — treat as changed with similarity 0.
                changed.append({
                    "article_id": art_id,
                    "old_content": "",
                    "new_content": new_content,
                    "old_hash": None,
                    "new_hash": new_hash,
                    "cosine_similarity": 0.0,
                })
                continue

            old_hash = stored_article["content_hash"]

            # Fast path: identical hashes mean identical content.
            if old_hash == new_hash:
                continue

            sim = semantic_diff(stored_article["content"], new_content)

            if sim < self._threshold:
                changed.append({
                    "article_id": art_id,
                    "old_content": stored_article["content"],
                    "new_content": new_content,
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                    "cosine_similarity": round(sim, 4),
                })

        if changed:
            logger.info(
                "Detected %d changed article(s) for '%s' (threshold=%.2f).",
                len(changed), regulation, self._threshold,
            )
        else:
            logger.info("No meaningful changes detected for '%s'.", regulation)

        return changed

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply_updates(
        self,
        regulation: str,
        changed_articles: list[dict],
    ) -> None:
        """Re-embed changed articles, upsert to ChromaDB, log changes, and
        flag stale documents.

        Parameters
        ----------
        regulation : str
            Regulation namespace.
        changed_articles : list[dict]
            Output of :meth:`check_for_updates`.
        """
        if not changed_articles:
            return

        for article in changed_articles:
            art_id = article["article_id"]
            new_content = article["new_content"]
            new_hash = article["new_hash"]
            old_hash = article.get("old_hash")
            sim = article.get("cosine_similarity")

            # 1. Re-embed the updated article.
            new_embedding = embedder.embed(new_content)

            # 2. Upsert into ChromaDB.
            vector_store.upsert(
                namespace=regulation,
                ids=[art_id],
                documents=[new_content],
                embeddings=[new_embedding],
                metadatas=[{
                    "article_id": art_id,
                    "regulation": regulation,
                    "content_hash": new_hash,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }],
            )

            # 3. Log the change in the regulation changelog.
            changelog_id = regulation_changelog.log_change(
                regulation=regulation,
                article_id=art_id,
                old_hash=old_hash,
                new_hash=new_hash,
                similarity_score=sim,
            )

            # 4. Mark the changelog entry as reindexed.
            regulation_changelog.mark_reindexed(changelog_id)

            # 5. Flag documents evaluated against the old version.
            affected = regulation_changelog.flag_documents_for_reevaluation(
                regulation=regulation,
                article_id=art_id,
                new_hash=new_hash,
            )

            if affected:
                regulation_changelog.set_affected_doc_ids(changelog_id, affected)
                logger.info(
                    "Flagged %d document(s) for re-evaluation on %s/%s.",
                    len(affected), regulation, art_id,
                )

        logger.info(
            "Applied %d update(s) for '%s'.", len(changed_articles), regulation,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
regulation_watcher = RegulationWatcher()
