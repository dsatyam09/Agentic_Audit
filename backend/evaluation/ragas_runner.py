"""RAGASRunner: Faithfulness + Answer Relevance evaluation.

Targets:
- Faithfulness ≥ 0.80
- Answer Relevance ≥ 0.75
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import subprocess
import sys
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline


class _ChatHuggingFacePipelineAsyncShim(ChatHuggingFace):
    """RAGAS uses agenerate_prompt; ChatHuggingFace rejects async for HuggingFacePipeline."""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(self.llm, HuggingFacePipeline):
            return await asyncio.to_thread(
                self._generate,
                messages,
                stop,
                None,
                stream,
                **kwargs,
            )
        return await super()._agenerate(messages, stop, run_manager, stream, **kwargs)


class RAGASRunner:
    @staticmethod
    def _safe_score(result, key: str):
        """Aggregate per-row metric using nanmean (partial successes still count)."""
        import math

        try:
            df = result.to_pandas()
            if key not in df.columns:
                return None, {"total": 0, "valid": 0}
            col = df[key]
            total = int(len(col))
            valid_mask = col.apply(
                lambda v: v is not None and isinstance(v, (int, float)) and not (
                    isinstance(v, float) and (math.isnan(v) or math.isinf(v))
                )
            )
            valid = int(valid_mask.sum())
            if valid == 0:
                return None, {"total": total, "valid": 0}
            mean_val = float(col[valid_mask].astype(float).mean())
            return round(mean_val, 4), {"total": total, "valid": valid}
        except Exception:
            try:
                v = result[key]
                if v is None:
                    return None, {"total": 0, "valid": 0}
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    return None, {"total": 0, "valid": 0}
                return round(fv, 4), {"total": 1, "valid": 1}
            except Exception:
                return None, {"total": 0, "valid": 0}

    @staticmethod
    def _describe_result(result) -> str:
        """Per-row snapshot for debugging null scores."""
        try:
            df = result.to_pandas()
            keep = [c for c in df.columns if c in ("faithfulness", "answer_relevancy")]
            if not keep:
                return ""
            view = df[keep].head(5).to_dict(orient="records")
            return f"per_row_scores(sample)={view}"
        except Exception as e:  # noqa: BLE001
            return f"(result inspection failed: {e})"

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

    @staticmethod
    def _get_hf_ragas_model() -> str:
        """Model id used for Hugging Face-backed RAGAS evaluation."""
        # Override with HF_RAGAS_MODEL; fallback tuned for HF Inference providers.
        return os.environ.get("HF_RAGAS_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    @staticmethod
    def _get_hf_provider() -> str:
        return os.environ.get("HF_RAGAS_PROVIDER", "")

    @staticmethod
    def _get_hf_embed_provider() -> str:
        return os.environ.get("HF_RAGAS_EMBED_PROVIDER", "hf-inference")

    @staticmethod
    def _get_hf_task() -> str:
        # Many hosted providers expose chat/instruct models as "conversational".
        return os.environ.get("HF_RAGAS_TASK", "conversational")

    def _build_hf_llm_wrapper(self):
        """Build a LangchainLLMWrapper around HuggingFaceHub, if configured."""
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return None, "HF_TOKEN missing"
        try:
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
            from ragas.llms import LangchainLLMWrapper

            provider = self._get_hf_provider()
            llm_kwargs = {
                "model": self._get_hf_ragas_model(),
                "huggingfacehub_api_token": hf_token,
                "temperature": 0.0,
                "max_new_tokens": 512,
                "do_sample": False,
                "task": self._get_hf_task(),
            }
            if provider:
                llm_kwargs["provider"] = provider
            base_llm = HuggingFaceEndpoint(**llm_kwargs)
            chat_llm = ChatHuggingFace(llm=base_llm, model_id=self._get_hf_ragas_model())
            return LangchainLLMWrapper(chat_llm), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    @staticmethod
    def _get_hf_embed_model() -> str:
        return os.environ.get(
            "HF_RAGAS_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    @staticmethod
    def _get_local_ragas_model() -> str:
        """Non-empty when local transformers weights should be used for RAGAS."""
        return os.environ.get("LOCAL_RAGAS_MODEL", "").strip()

    @staticmethod
    def _get_local_embed_model() -> str:
        return os.environ.get(
            "LOCAL_RAGAS_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip()

    def _build_local_llm_wrapper(self):
        """Local HuggingFace pipeline + ChatHuggingFace (no Inference API / router)."""
        model_id = self._get_local_ragas_model()
        if not model_id:
            return None, "LOCAL_RAGAS_MODEL empty"
        try:
            from ragas.llms import LangchainLLMWrapper

            hf_token = os.environ.get("HF_TOKEN")
            model_kwargs: dict = {}
            if hf_token:
                model_kwargs["token"] = hf_token
            if os.environ.get("LOCAL_RAGAS_TRUST_REMOTE_CODE", "true").lower() in (
                "1",
                "true",
                "yes",
            ):
                model_kwargs["trust_remote_code"] = True

            dtype_name = os.environ.get("LOCAL_RAGAS_DTYPE", "bfloat16").strip().lower()
            try:
                import torch

                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }
                if dtype_name in dtype_map:
                    model_kwargs["dtype"] = dtype_map[dtype_name]
            except Exception:  # noqa: BLE001
                pass

            do_sample = os.environ.get("LOCAL_RAGAS_DO_SAMPLE", "true").lower() in (
                "1",
                "true",
                "yes",
            )
            temperature = float(os.environ.get("LOCAL_RAGAS_TEMPERATURE", "0.7"))
            top_p = float(os.environ.get("LOCAL_RAGAS_TOP_P", "0.9"))
            pipeline_kwargs: dict = {
                "max_new_tokens": int(os.environ.get("LOCAL_RAGAS_MAX_NEW_TOKENS", "384")),
                "do_sample": do_sample,
            }
            if do_sample:
                pipeline_kwargs["temperature"] = temperature
                pipeline_kwargs["top_p"] = top_p
                pipeline_kwargs["return_full_text"] = False

            pipeline_extra: dict = {}
            dm = os.environ.get("LOCAL_RAGAS_DEVICE_MAP", "").strip()
            if dm:
                pipeline_extra["device_map"] = dm
            dev_raw = os.environ.get("LOCAL_RAGAS_DEVICE", "").strip()
            if dev_raw and not dm:
                try:
                    pipeline_extra["device"] = int(dev_raw)
                except ValueError:
                    pass

            chat = _ChatHuggingFacePipelineAsyncShim.from_model_id(
                model_id=model_id,
                task="text-generation",
                backend="pipeline",
                model_kwargs=model_kwargs or None,
                pipeline_kwargs=pipeline_kwargs,
                **pipeline_extra,
            )
            return LangchainLLMWrapper(chat), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    def _build_llm_factory_llm(self):
        """Modern ragas path: llm_factory(AsyncOpenAI(...)) via Instructor adapter.

        Recommended pattern from https://github.com/vibrantlabsai/ragas (replaces the
        deprecated LangchainLLMWrapper). Works with any OpenAI-compatible endpoint,
        including local Ollama (`OLLAMA_BASE_URL=http://localhost:11434/v1`).
        """
        try:
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
        except Exception as e:  # noqa: BLE001
            return None, f"openai/ragas import failed: {e}", None, None

        base_url = (
            os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("RAGAS_OPENAI_BASE_URL")
            or ""
        ).strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if base_url:
            api_key = os.environ.get("OLLAMA_API_KEY", "ollama").strip() or "ollama"
            model = (
                os.environ.get("OLLAMA_RAGAS_MODEL")
                or os.environ.get("RAGAS_OPENAI_MODEL")
                or "qwen2.5:3b-instruct"
            ).strip()
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            provider_label = f"openai-compatible:{base_url}:{model}"
        elif openai_key:
            model = os.environ.get("RAGAS_OPENAI_MODEL", "gpt-4o-mini").strip()
            client = AsyncOpenAI(api_key=openai_key)
            provider_label = f"openai:{model}"
        else:
            return None, "no OpenAI-compatible endpoint configured", None, None

        try:
            llm = llm_factory(model, provider="openai", client=client)
        except Exception as e:  # noqa: BLE001
            return None, f"llm_factory failed: {e}", None, None
        return llm, None, client, provider_label

    def _build_local_embeddings_wrapper(self):
        """sentence-transformers via langchain_huggingface (local, no HF router)."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper

            model_name = self._get_local_embed_model() or self._get_hf_embed_model()
            mk: dict = {}
            tok = os.environ.get("HF_TOKEN")
            if tok:
                mk["token"] = tok
            ed = os.environ.get("LOCAL_RAGAS_EMBED_DEVICE", "").strip()
            if ed:
                mk["device"] = ed

            emb = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=mk)
            return LangchainEmbeddingsWrapper(emb), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    def _build_hf_embeddings_wrapper(self):
        """Build a LangchainEmbeddingsWrapper using Hugging Face endpoint embeddings."""
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return None, "HF_TOKEN missing"
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper

            provider = self._get_hf_embed_provider()
            emb_kwargs = {
                "model": self._get_hf_embed_model(),
                "huggingfacehub_api_token": hf_token,
                "task": "feature-extraction",
            }
            if provider:
                emb_kwargs["provider"] = provider
            emb = HuggingFaceEndpointEmbeddings(**emb_kwargs)
            return LangchainEmbeddingsWrapper(emb), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    def _evaluate_in_subprocess(self, data: dict) -> dict:
        """Run ragas in a separate Python process to avoid uvloop nesting issues."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_subprocess.py")
        env = {**os.environ, "PYTHONPATH": project_root}
        proc = subprocess.run(
            [sys.executable, script],
            input=json.dumps(data),
            text=True,
            capture_output=True,
            check=False,
            cwd=project_root,
            env=env,
        )
        stdout = (proc.stdout or "").strip()
        if not stdout:
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "error": proc.stderr.strip() or "ragas subprocess produced no output",
            }
        try:
            return json.loads(stdout.splitlines()[-1])
        except Exception:
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "error": f"invalid ragas subprocess output: {stdout[:300]}",
            }

    def _evaluate_dataset(self, data: dict) -> dict:
        """Core RAGAS run: pick backends, evaluate, return scores dict."""
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy]
        llm_wrapper = None
        embeddings_wrapper = None
        provider = "default"

        # Preferred path (ragas 0.4.x): llm_factory with OpenAI-compatible client.
        # Works with real OpenAI or a local Ollama server — the Instructor adapter
        # handles structured output retries that LangchainLLMWrapper lacks.
        use_factory = (
            os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("RAGAS_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_KEY")
        )
        if use_factory:
            factory_llm, ferr, fclient, flabel = self._build_llm_factory_llm()
            if factory_llm is None:
                return {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "error": f"Failed to initialize RAGAS llm_factory: {ferr}",
                }
            llm_wrapper = factory_llm
            # Pair with local sentence-transformers embeddings (fast, no API cost).
            embeddings_wrapper, emb_err = self._build_local_embeddings_wrapper()
            if embeddings_wrapper is None:
                return {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "error": f"Failed to initialize embeddings: {emb_err}",
                }
            provider = flabel
        elif self._get_local_ragas_model():
            llm_wrapper, err = self._build_local_llm_wrapper()
            if llm_wrapper is None:
                return {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "error": f"Failed to initialize local RAGAS LLM: {err}",
                }
            embeddings_wrapper, emb_err = self._build_local_embeddings_wrapper()
            if embeddings_wrapper is None:
                return {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "error": f"Failed to initialize local RAGAS embeddings: {emb_err}",
                }
            provider = (
                f"local:llm={self._get_local_ragas_model()};"
                f"emb={self._get_local_embed_model()}"
            )
        elif os.environ.get("HF_TOKEN"):
            llm_wrapper, err = self._build_hf_llm_wrapper()
            if llm_wrapper is None and err:
                return {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "error": f"Failed to initialize HF RAGAS LLM: {err}",
                }
            embeddings_wrapper, emb_err = self._build_hf_embeddings_wrapper()
            if embeddings_wrapper is None and emb_err:
                return {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "error": f"Failed to initialize HF embeddings for RAGAS: {emb_err}",
                }
            provider = (
                f"huggingface:llm={self._get_hf_provider() or 'auto'}:{self._get_hf_ragas_model()};"
                f"emb={self._get_hf_embed_provider() or 'auto'}:{self._get_hf_embed_model()}"
            )
        else:
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "error": (
                    "No RAGAS backend: set OPENAI_API_KEY, or LOCAL_RAGAS_MODEL for local "
                    "weights, or HF_TOKEN for hosted inference."
                ),
            }

        eval_error: str | None = None
        result = None
        try:
            result = ragas_evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm_wrapper,
                embeddings=embeddings_wrapper,
                raise_exceptions=False,
                show_progress=False,
            )
        except Exception as e:  # noqa: BLE001
            eval_error = f"ragas_evaluate raised: {e}"

        if result is not None:
            faith, faith_cov = self._safe_score(result, "faithfulness")
            rel, rel_cov = self._safe_score(result, "answer_relevancy")
        else:
            faith, rel = None, None
            faith_cov = rel_cov = {"total": 0, "valid": 0}

        out: dict = {
            "faithfulness": faith,
            "answer_relevancy": rel,
            "provider": provider,
            "coverage": {
                "faithfulness": faith_cov,
                "answer_relevancy": rel_cov,
            },
        }
        if faith is None and rel is None:
            detail = self._describe_result(result) if result is not None else ""
            msg = (
                eval_error
                or "RAGAS produced no valid rows — the LLM likely failed to emit "
                "valid structured output for every metric. Try a stronger "
                "OLLAMA_RAGAS_MODEL (e.g. qwen2.5:7b-instruct) or set "
                "OPENAI_API_KEY."
            )
            out["error"] = (msg + (f" | {detail}" if detail else ""))[:800]
            print(f"[ragas] {out['error']}", file=sys.stderr, flush=True)
        elif faith is None or rel is None:
            missing = "faithfulness" if faith is None else "answer_relevancy"
            detail = self._describe_result(result) if result is not None else ""
            msg = (
                f"{missing} produced NaN for all rows — model could not satisfy "
                f"that metric's JSON schema. Other metric computed successfully."
            )
            out["error"] = (msg + (f" | {detail}" if detail else ""))[:800]
            print(f"[ragas] {out['error']}", file=sys.stderr, flush=True)
        return out

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

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        try:
            # RAGAS calls asyncio.run() internally. Under FastAPI/uvicorn there is already
            # a running loop on the request thread, which breaks asyncio.run(). Run the
            # whole evaluation in a worker thread that has no active loop.
            timeout_sec = int(os.environ.get("RAGAS_EVAL_TIMEOUT_SEC", "3600"))

            def _run_ragas_in_worker() -> dict:
                return self._evaluate_dataset(data)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_run_ragas_in_worker).result(timeout=timeout_sec)
        except Exception as e:
            # Known issue under uvicorn/uvloop: nested async loop execution in ragas.
            msg = str(e).lower()
            if "nested async" in msg or "uvloop" in msg or "event loop" in msg:
                return self._evaluate_in_subprocess(data)
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
