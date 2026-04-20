"""Generate synthetic enterprise documents for testing.

Usage:
    python scripts/generate_docs.py --regulation gdpr --count 5
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test documents")
    parser.add_argument("--regulation", required=True, choices=["gdpr", "hipaa", "nist"])
    parser.add_argument("--count", type=int, default=5, help="Number of docs per category")
    args = parser.parse_args()

    print(f"Document generation for {args.regulation}")
    print("This script requires manual creation or LLM-assisted generation.")
    print(f"Place documents in: data/testing/documents/{args.regulation}/")
    print(f"  - compliant/     (comp_001.pdf ... comp_{args.count:03d}.pdf)")
    print(f"  - non_compliant/ (nc_001.pdf ... nc_{args.count:03d}.pdf)")
    print(f"  - ambiguous/     (amb_001.pdf ... amb_{args.count:03d}.pdf)")
    print()
    print("Existing test documents in test_datasets/ can be symlinked to the spec paths.")


if __name__ == "__main__":
    main()
