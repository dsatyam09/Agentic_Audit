"""DebateAgent: LangGraph node that orchestrates the adversarial debate for all (chunk, clause) pairs.

Calls DebateProtocol.run_debate for each pair and collects DebateRecords.
"""

from backend.agents.state import ComplianceState, DebateRecord
from backend.debate.protocol import run_debate
from backend.debate.qwen_runner import qwen
from backend.logging.pipeline_log import make_log_entry


def debate_node(state: ComplianceState) -> dict:
    """LangGraph node: runs adversarial debate for every (chunk, clause) pair."""
    retrieved_clauses = state["retrieved_clauses"]
    debate_records: list[DebateRecord] = []
    log_entries = []

    for chunk_data in retrieved_clauses:
        chunk_index = chunk_data["chunk_index"]
        chunk_text = chunk_data["chunk_text"]
        clauses = chunk_data.get("clauses", [])

        for clause in clauses:
            record = run_debate(
                chunk_text=chunk_text,
                clause=clause,
                chunk_index=chunk_index,
                qwen_runner=qwen,
            )
            debate_records.append(record)

            # Log each debate round
            log = make_log_entry(
                agent="debate",
                input_data={
                    "chunk_index": chunk_index,
                    "article_id": clause["article_id"],
                    "regulation": clause["regulation"],
                },
                raw_prompt=None,
                thinking_trace=record.get("arbiter_thinking", ""),
                raw_response=None,
                structured_output={
                    "verdict": record["verdict"],
                    "risk_level": record["risk_level"],
                    "hallucination_flag": record["hallucination_flag"],
                },
                article_id=clause["article_id"],
                regulation=clause["regulation"],
                chunk_index=chunk_index,
            )
            log_entries.append(log)

    return {
        "debate_records": debate_records,
        "pipeline_log": log_entries,
    }
