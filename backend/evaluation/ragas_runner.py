"""RAGASRunner: Faithfulness + Answer Relevance evaluation.

Targets:
- Faithfulness ≥ 0.80
- Answer Relevance ≥ 0.75
"""

import json
from pathlib import Path


class RAGASRunner:
    """Runs RAGAS evaluation metrics on pipeline outputs."""

    def __init__(self):
        self._ragas_available = None

    def _check_ragas(self) -> bool:
        if self._ragas_available is None:
            try:
                import ragas
                self._ragas_available = True
            except ImportError:
                self._ragas_available = False
        return self._ragas_available

    def evaluate(self, questions: list[str], answers: list[str],
                 contexts: list[list[str]], ground_truths: list[str] = None) -> dict:
        """
        Run RAGAS evaluation.

        Args:
            questions: The query/question for each evaluation
            answers: The model's answer for each evaluation
            contexts: Retrieved contexts for each evaluation
            ground_truths: Optional ground truth answers

        Returns:
            Dict with faithfulness, answer_relevancy scores
        """
        if not self._check_ragas():
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "error": "ragas not installed",
            }

        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import faithfulness, answer_relevancy
            from datasets import Dataset

            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            if ground_truths:
                data["ground_truth"] = ground_truths

            dataset = Dataset.from_dict(data)

            metrics = [faithfulness, answer_relevancy]
            result = ragas_evaluate(dataset=dataset, metrics=metrics)

            return {
                "faithfulness": round(result["faithfulness"], 4),
                "answer_relevancy": round(result["answer_relevancy"], 4),
            }
        except Exception as e:
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "error": str(e),
            }

    def evaluate_from_pipeline(self, state: dict) -> dict:
        """Build RAGAS inputs from a pipeline state and evaluate."""
        questions = []
        answers = []
        contexts = []

        debate_records = state.get("debate_records", [])
        retrieved_clauses = state.get("retrieved_clauses", [])

        # Build context lookup by chunk_index
        context_lookup = {}
        for chunk_data in retrieved_clauses:
            idx = chunk_data["chunk_index"]
            context_lookup[idx] = [c.get("clause_text", "") for c in chunk_data.get("clauses", [])]

        for record in debate_records:
            question = (
                f"Does the policy comply with {record['regulation'].upper()} "
                f"{record['article_id']} — {record['article_title']}?"
            )
            answer = record.get("reasoning", record.get("verdict", ""))
            ctx = context_lookup.get(record["chunk_index"], [])

            questions.append(question)
            answers.append(answer)
            contexts.append(ctx)

        if not questions:
            return {"faithfulness": None, "answer_relevancy": None, "error": "no records"}

        return self.evaluate(questions, answers, contexts)


ragas_runner = RAGASRunner()
