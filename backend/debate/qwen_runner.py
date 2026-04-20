"""QwenRunner — singleton wrapper around a Qwen model for compliance debate inference.

Handles VRAM detection, 4-bit quantization fallback, and structured output
with thinking-trace extraction.

The model ID is read from the QWEN_MODEL_ID environment variable, defaulting
to Qwen/Qwen3-8B.  For testing on CPU without downloading 8B weights, set:
    QWEN_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct

Use ``get_qwen()`` for lazy access; the old module-level singleton
``qwen`` is provided for backwards compatibility but backed by the lazy
loader to avoid loading the model at import time.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>(.*)", re.DOTALL | re.IGNORECASE)


def _is_qwen3(model_id: str) -> bool:
    mid = model_id.lower()
    return "qwen3" in mid or "qwen/qwen3" in mid


class QwenRunner:
    MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-8B")

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        try:
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory
                if vram < 16 * 1024**3:
                    from transformers import BitsAndBytesConfig

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.MODEL_ID,
                        quantization_config=bnb_config,
                        device_map="auto",
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.MODEL_ID,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
        self.model.eval()

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict], thinking: bool) -> str:
        """Apply the tokenizer's chat template with native thinking control when available."""
        # Qwen3 tokenizers support an ``enable_thinking`` kwarg that toggles
        # the native <think> reasoning channel. Older Qwen2.5 tokenizers do
        # not — we fall back to prompt-level instructions.
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        thinking: bool = True,
        max_new_tokens: int = 1024,
    ) -> dict:
        """Run a single inference pass on the loaded Qwen model.

        Returns
        -------
        dict with keys:
            thinking_trace : str — content between ``<think>...</think>`` tags
                (empty string when *thinking* is False or no tags present).
            response : str — text after ``</think>`` (or the full output when
                no tags are detected).
            full_output : str — complete raw model output.
        """
        if thinking:
            system_msg = (
                "You are an expert compliance auditor. "
                "Reason step-by-step inside <think>...</think> tags before answering. "
                "After </think> respond with ONLY the requested JSON object — no prose, "
                "no markdown fences, no additional commentary."
            )
        else:
            system_msg = (
                "You are an expert compliance auditor. "
                "Respond with ONLY the requested JSON object — no prose, "
                "no markdown fences, no additional commentary. Do not use <think> tags."
            )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        text = self._apply_chat_template(messages, thinking=thinking)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_output = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        thinking_trace, response = _split_thinking(full_output)

        # If thinking was disabled but the model still emitted tags, keep the
        # post-</think> portion as the response and leave thinking_trace in
        # case the caller wants to inspect it (it's still useful logging).
        if not thinking and thinking_trace:
            # Respect ablation request: don't expose thinking upstream.
            thinking_trace = ""

        return {
            "thinking_trace": thinking_trace,
            "response": response,
            "full_output": full_output,
        }


def _split_thinking(full_output: str) -> tuple[str, str]:
    """Extract ``<think>...</think>`` trace and post-think response from *full_output*.

    Handles three shapes the model may emit:
    1. ``<think>reasoning</think> response`` — balanced tags.
    2. ``reasoning</think> response`` — open tag dropped (common with some Qwen3 templates).
    3. ``<think>reasoning`` — closing tag missing (truncation).
    4. No tags at all — return ("", full_output).
    """
    if not isinstance(full_output, str):
        return "", ""

    match = _THINK_RE.search(full_output)
    if match:
        thinking_trace = match.group(1).strip()
        # response is everything *after* the matched </think>
        response = full_output[match.end() :].strip()
        return thinking_trace, response

    if "</think>" in full_output:
        # Template emitted the closing tag without an explicit opening tag
        # (Qwen3 chat template sometimes injects <think> implicitly and we
        # only see the closing tag in the decoded output).
        pre, _, post = full_output.partition("</think>")
        return pre.strip(), post.strip()

    open_match = _OPEN_THINK_RE.search(full_output)
    if open_match:
        # Truncated before closing tag — treat whatever was emitted as trace,
        # and the response falls back to the full output minus the trace.
        return open_match.group(1).strip(), full_output.strip()

    return "", full_output.strip()


# ---------------------------------------------------------------------------
# Lazy module-level access
# ---------------------------------------------------------------------------

_runner: QwenRunner | None = None
_runner_lock = threading.Lock()


def get_qwen() -> QwenRunner:
    """Return the process-wide QwenRunner singleton, loading it on first use."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = QwenRunner()
    return _runner


class _LazyQwenProxy:
    """Attribute-forwarding proxy used for backwards compatibility with
    ``from backend.debate.qwen_runner import qwen``.

    The real ``QwenRunner`` is not constructed until the first attribute
    access, which keeps imports cheap and lets test code override
    ``QWEN_MODEL_ID`` without side effects.
    """

    def __getattr__(self, item: str) -> Any:
        return getattr(get_qwen(), item)

    def __repr__(self) -> str:  # pragma: no cover — trivial
        return "<LazyQwenProxy>"


qwen = _LazyQwenProxy()
