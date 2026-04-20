"""LangGraph StateGraph: wires all agents into a pipeline.

Pipeline:
    Classifier → Retrieval → Debate → Reporter(compute)
                                          → (optional) Drift
                                              → Reporter(render) → END

The reporter is split so drift analysis runs *before* PDF rendering, which
means the Assessment/Remediation reports always include the latest drift
table when a previous report is available.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from langgraph.graph import StateGraph, END

from backend.agents.state import ComplianceState
from backend.agents.classifier import classifier_node
from backend.agents.retrieval_agent import retrieval_node
from backend.agents.debate_agent import debate_node
from backend.agents.reporter import reporter_compute_node, reporter_render_node
from backend.drift.detector import drift_node


def build_graph() -> StateGraph:
    graph = StateGraph(ComplianceState)

    graph.add_node("classifier", classifier_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("debate", debate_node)
    graph.add_node("reporter_compute", reporter_compute_node)
    graph.add_node("drift", drift_node)
    graph.add_node("reporter_render", reporter_render_node)

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "retrieval")
    graph.add_edge("retrieval", "debate")
    graph.add_edge("debate", "reporter_compute")

    # Drift only runs when a previous report is supplied. Either way, we
    # continue to the render node so PDFs include drift data when present.
    graph.add_conditional_edges(
        "reporter_compute",
        lambda state: "drift" if state.get("previous_report_path") else "reporter_render",
        {"drift": "drift", "reporter_render": "reporter_render"},
    )
    graph.add_edge("drift", "reporter_render")
    graph.add_edge("reporter_render", END)

    return graph.compile()


compiled_graph = build_graph()


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(
    doc_path: str,
    previous_report_path: str | None = None,
    thinking: bool = True,
) -> ComplianceState:
    """Run the full compliance evaluation pipeline on a document.

    Parameters
    ----------
    doc_path:
        Filesystem path to the enterprise document.
    previous_report_path:
        Optional path to a prior ``violation_report.json`` for drift detection.
    thinking:
        When False, disables Qwen ``<think>`` traces across debate and
        remediation (the C4-nothink ablation condition).
    """
    from backend.ingestion.parser import DocumentParser
    from backend.ingestion.chunker import DocumentChunker
    from backend.logging.pipeline_log import flush_pipeline_log

    abs_doc_path = str(Path(doc_path).resolve())
    doc_text = DocumentParser().parse(abs_doc_path)
    doc_chunks = DocumentChunker().chunk(doc_text)

    # Short, stable identifier derived from the path (used for directory layout).
    doc_id = hashlib.sha256(abs_doc_path.encode()).hexdigest()[:12]
    # Content hash is distinct — lets drift detection and changelog track
    # *what* was evaluated, not just where it lived on disk.
    doc_sha256 = _sha256_of_file(abs_doc_path)
    doc_filename = Path(abs_doc_path).name

    initial_state: ComplianceState = {
        "doc_id": doc_id,
        "doc_path": abs_doc_path,
        "doc_filename": doc_filename,
        "doc_sha256": doc_sha256,
        "doc_text": doc_text,
        "doc_chunks": doc_chunks,
        "thinking_enabled": thinking,
        "doc_type": "",
        "regulation_scope": [],
        "classifier_confidence": 0.0,
        "classifier_reasoning": "",
        "retrieved_clauses": [],
        "debate_records": [],
        "risk_score": 0.0,
        "risk_level": "Low",
        "violation_report": {},
        "poam": {"assessment_report_path": "", "remediation_report_path": ""},
        "previous_report_path": previous_report_path,
        "drift_result": None,
        "pipeline_log": [],
    }

    final_state = compiled_graph.invoke(initial_state)

    # Persist the pipeline log. The violation_report itself is written by
    # reporter_compute_node so drift + render can read it back from disk.
    flush_pipeline_log(final_state["pipeline_log"], doc_id)

    return final_state
