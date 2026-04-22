"""LangGraph StateGraph: wires all agents into a pipeline.

Pipeline: Classifier → Retrieval → Debate → Reporter → (conditional) Drift → END
"""

from langgraph.graph import StateGraph, END
from backend.agents.state import ComplianceState
from backend.agents.classifier import classifier_node
from backend.agents.retrieval_agent import retrieval_node
from backend.agents.debate_agent import debate_node
from backend.agents.reporter import reporter_node
from backend.drift.detector import drift_node


def build_graph() -> StateGraph:
    graph = StateGraph(ComplianceState)

    graph.add_node("classifier", classifier_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("debate", debate_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("drift", drift_node)

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "retrieval")
    graph.add_edge("retrieval", "debate")
    graph.add_edge("debate", "reporter")

    # Drift node only runs if previous_report_path is provided
    graph.add_conditional_edges(
        "reporter",
        lambda state: "drift" if state.get("previous_report_path") else END,
        {"drift": "drift", END: END},
    )
    graph.add_edge("drift", END)

    return graph.compile()


compiled_graph = build_graph()


def run_pipeline(doc_path: str, previous_report_path: str | None = None) -> ComplianceState:
    """Run the full compliance evaluation pipeline on a document."""
    from backend.ingestion.parser import DocumentParser
    from backend.ingestion.chunker import DocumentChunker
    from backend.logging.pipeline_log import flush_pipeline_log
    import hashlib
    import os

    doc_text = DocumentParser().parse(doc_path)
    doc_chunks = DocumentChunker().chunk(doc_text)
    doc_id = hashlib.sha256(doc_path.encode()).hexdigest()[:12]

    initial_state = ComplianceState(
        doc_id=doc_id,
        doc_path=doc_path,
        doc_text=doc_text,
        doc_chunks=doc_chunks,
        doc_type="",
        regulation_scope=[],
        classifier_confidence=0.0,
        classifier_reasoning="",
        retrieved_clauses=[],
        debate_records=[],
        risk_score=0.0,
        risk_level="Low",
        violation_report={},
        poam={"assessment_report_path": "", "remediation_report_path": ""},
        previous_report_path=previous_report_path,
        drift_result=None,
        pipeline_log=[],
    )

    final_state = compiled_graph.invoke(initial_state)

    # Write outputs
    flush_pipeline_log(final_state["pipeline_log"], doc_id)

    report_path = f"outputs/reports/{doc_id}/raw/violation_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    import json
    with open(report_path, "w") as f:
        json.dump(final_state["violation_report"], f, indent=2)

    return final_state
