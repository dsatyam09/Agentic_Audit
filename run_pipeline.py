"""CLI entry point: evaluate a single document.

Usage:
    python run_pipeline.py --doc data/testing/documents/gdpr/non_compliant/nc_001.pdf
    python run_pipeline.py --doc path/to/doc.pdf --previous-report outputs/reports/abc123/raw/violation_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Run compliance evaluation pipeline on a single document"
    )
    parser.add_argument(
        "--doc", required=True,
        help="Path to the enterprise document (PDF, DOCX, or TXT)"
    )
    parser.add_argument(
        "--previous-report", default=None,
        help="Path to a previous violation_report.json for drift detection"
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable Qwen <think> reasoning (C4-nothink ablation condition)",
    )
    args = parser.parse_args()

    doc_path = str(Path(args.doc).resolve())
    if not Path(doc_path).exists():
        print(f"Error: Document not found: {doc_path}")
        sys.exit(1)

    previous_report_path = None
    if args.previous_report:
        previous_report_path = str(Path(args.previous_report).resolve())
        if not Path(previous_report_path).exists():
            print(f"Error: Previous report not found: {previous_report_path}")
            sys.exit(1)

    thinking = not args.no_thinking

    print(f"Starting compliance evaluation pipeline...")
    print(f"  Document: {doc_path}")
    print(f"  Thinking enabled: {thinking}")
    if previous_report_path:
        print(f"  Previous report: {previous_report_path} (drift detection enabled)")
    print()

    from backend.graph import run_pipeline
    final_state = run_pipeline(doc_path, previous_report_path, thinking=thinking)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Document ID:      {final_state['doc_id']}")
    print(f"  Filename:         {final_state.get('doc_filename', '—')}")
    print(f"  Document Type:    {final_state['doc_type']}")
    print(f"  Regulations:      {', '.join(final_state['regulation_scope'])}")
    print(f"  Risk Score:       {final_state['risk_score']} / 4.0")
    print(f"  Risk Level:       {final_state['risk_level']}")

    vr = final_state.get("violation_report", {})
    violations = vr.get("violations", [])
    n_missing = sum(1 for v in violations if v["verdict"] == "Missing")
    n_partial = sum(1 for v in violations if v["verdict"] == "Partial")
    n_full = sum(1 for v in violations if v["verdict"] == "Full")
    print(f"  Articles Evaluated: {vr.get('articles_evaluated', 0)}")
    print(f"  Full: {n_full}  Partial: {n_partial}  Missing: {n_missing}")
    print(f"  Hallucination Rate: {vr.get('hallucination_rate', 0):.1%}")

    poam = final_state.get("poam", {})
    if poam.get("assessment_report_path"):
        print(f"\n  Assessment Report: {poam['assessment_report_path']}")
    if poam.get("remediation_report_path"):
        print(f"  Remediation Report: {poam['remediation_report_path']}")

    if final_state.get("drift_result"):
        dr = final_state["drift_result"]
        print(f"\n  DRIFT DETECTED:")
        print(f"    Risk Score Delta: {dr['risk_score_delta']}")
        print(f"    Regressions: {dr['regression_count']}")
        if dr.get("critical_regressions"):
            print(f"    Critical Regressions: {', '.join(dr['critical_regressions'])}")

    report_path = f"outputs/reports/{final_state['doc_id']}/raw/violation_report.json"
    print(f"\n  Full report: {report_path}")


if __name__ == "__main__":
    main()
