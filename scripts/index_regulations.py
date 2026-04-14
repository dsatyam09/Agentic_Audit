"""Embed articles.json → Chroma (run once per regulation).

Usage:
    python scripts/index_regulations.py --regulation gdpr
    python scripts/index_regulations.py --regulation soc2
    python scripts/index_regulations.py --regulation hipaa
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def main():
    parser = argparse.ArgumentParser(description="Index regulation articles into Chroma")
    parser.add_argument("--regulation", required=True,
                        choices=["gdpr", "soc2", "hipaa", "iso27001"])
    args = parser.parse_args()
    index_regulation(args.regulation)


if __name__ == "__main__":
    main()
