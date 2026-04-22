"""Embed articles.json → Chroma (run once per regulation).

Usage:
    python scripts/index_regulations.py --regulation gdpr
    python scripts/index_regulations.py --regulation hipaa
    python scripts/index_regulations.py --regulation nist
    python scripts/index_regulations.py --all
"""

import json
import argparse
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# So HF_TOKEN and other vars from the repo root `.env` apply (same as run_pipeline.py).
load_dotenv(PROJECT_ROOT / ".env")

# Install Hub auth + suppress spurious X-HF-Warning logs before any model import.
import backend.hf_setup  # noqa: F401


def index_regulation(regulation: str) -> None:
    """Load enriched articles and upsert to Chroma vector store."""
    from backend.retrieval.embedder import embedder
    from backend.retrieval.vector_store import vector_store

    articles_path = PROJECT_ROOT / "data" / "compliance" / regulation / f"{regulation}_articles.json"
    if not articles_path.exists():
        print(f"Error: {articles_path} not found. Run prepare_dataset.py first.")
        return

    with open(articles_path) as f:
        articles = json.load(f)

    if not articles:
        print(f"No articles found in {articles_path}.")
        return

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for article in articles:
        # Combine content + recital context for richer embedding
        text = article["content"]
        if article.get("recital_context"):
            text += "\n\n" + article["recital_context"]

        embedding = embedder.embed(text)

        ids.append(article["article_id"])
        documents.append(text)
        embeddings.append(embedding)
        metadatas.append({
            "article_id": article["article_id"],
            "article_title": article["article_title"],
            "regulation": regulation,
            "severity": article["severity"],
            "article_number": article.get("article_number", 0),
            "source_url": article.get("source_url", ""),
        })

    vector_store.upsert(
        namespace=regulation,
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Indexed {len(articles)} articles into Chroma namespace '{regulation}'")
    print(f"  Collection size: {vector_store.collection_size(regulation)}")


SUPPORTED_REGULATIONS = ["gdpr", "hipaa", "nist"]


def main():
    parser = argparse.ArgumentParser(description="Index regulation articles into Chroma")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--regulation", choices=SUPPORTED_REGULATIONS,
                       help="Index a single regulation namespace")
    group.add_argument("--all", action="store_true",
                       help="Index all supported regulations")
    args = parser.parse_args()

    targets = SUPPORTED_REGULATIONS if args.all else [args.regulation]
    for reg in targets:
        index_regulation(reg)


if __name__ == "__main__":
    main()
