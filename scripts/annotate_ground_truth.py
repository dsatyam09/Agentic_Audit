"""CLI annotation helper for creating ground truth annotations.

Usage:
    python scripts/annotate_ground_truth.py --regulation gdpr --doc-id nc_001
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from backend.agents.classifier import REGULATION_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["Full", "Partial", "Missing"]


def main():
    parser = argparse.ArgumentParser(description="Create ground truth annotations")
    parser.add_argument("--regulation", required=True, choices=list(REGULATION_REGISTRY.keys()))
    parser.add_argument("--doc-id", required=True, help="Document ID (e.g., nc_001)")
    parser.add_argument("--doc-path", default=None, help="Path to the document")
    args = parser.parse_args()

    gt_path = PROJECT_ROOT / "data" / "testing" / "ground_truth" / f"{args.regulation}_annotations.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if gt_path.exists():
        with open(gt_path) as f:
            existing = json.load(f)

    articles = REGULATION_REGISTRY[args.regulation]["focus_articles"]

    print(f"Annotating {args.doc_id} against {args.regulation}")
    print("Labels: Full | Partial | Missing")
    print()

    annotations = {}
    for art_id in articles:
        while True:
            label = input(f"  {art_id}: ").strip()
            if label in LABELS:
                notes = input("    Notes: ").strip()
                annotations[art_id] = {"label": label, "notes": notes}
                break
            print(f"    Invalid. Choose from: {', '.join(LABELS)}")

    entry = {
        "doc_id": args.doc_id,
        "doc_path": args.doc_path or f"data/testing/documents/{args.regulation}/{args.doc_id}.pdf",
        "regulation": args.regulation,
        "doc_type": "privacy_policy",
        "annotations": annotations,
        "annotation_sources": ["manual"],
        "agreed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    found = False
    for i, e in enumerate(existing):
        if e["doc_id"] == args.doc_id:
            existing[i] = entry
            found = True
            break
    if not found:
        existing.append(entry)

    with open(gt_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nSaved to {gt_path}")


if __name__ == "__main__":
    main()
