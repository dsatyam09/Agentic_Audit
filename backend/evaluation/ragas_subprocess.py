"""Run RAGAS in an isolated process (nested event loop / uvloop fallback)."""

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> None:
    from backend.evaluation.ragas_runner import ragas_runner

    data = json.loads(sys.stdin.read())
    out = ragas_runner._evaluate_dataset(data)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
