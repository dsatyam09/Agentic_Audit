"""QwenRunner — local/remote LLM wrapper for compliance debate inference.

Execution modes:
1) Remote OpenAI-compatible API (preferred on low-RAM hosts like Render Free)
2) Local Hugging Face model fallback
"""

import os

import backend.hf_setup  # noqa: F401 — load `.env` + HF auth before Hub access
from backend.hf_setup import hub_auth_token

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenRunner:
    MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-8B")

    def __init__(self):
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or ""
        ).strip()

        # OpenAI-compatible remote config
        self._remote_base_url = (
            os.environ.get("REMOTE_LLM_BASE_URL")
            or os.environ.get("GROQ_BASE_URL")
            or ""
        ).strip()
        self._remote_api_key = (
            os.environ.get("REMOTE_LLM_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or ""
        ).strip()
        self._remote_model = (
            os.environ.get("REMOTE_LLM_MODEL")
            or os.environ.get("GROQ_MODEL")
            or os.environ.get("HF_REMOTE_MODEL")
            or "Qwen/Qwen2.5-7B-Instruct"
        ).strip()
        self._remote_timeout = float(os.environ.get("REMOTE_LLM_TIMEOUT_SEC", "120"))
        self._remote_client = None
        # Allow explicitly disabling remote OpenAI-compatible providers so the
        # runner always uses locally downloaded HF weights.
        remote_enabled_raw = os.environ.get("REMOTE_LLM_ENABLED", "true").strip().lower()
        self._remote_enabled = remote_enabled_raw in {"1", "true", "yes", "on"}

        # Convenience path: if only HF_TOKEN is set, use HF's OpenAI-compatible router.
        if (
            self._remote_enabled
            and not self._remote_base_url
            and not self._remote_api_key
            and hf_token
        ):
            self._remote_base_url = "https://router.huggingface.co/v1"
            self._remote_api_key = hf_token

        self._use_remote = (
            self._remote_enabled
            and bool(self._remote_base_url and self._remote_api_key)
        )

        self.tokenizer = None
        self.model = None
        if not self._use_remote:
            self._init_local_model()

    def _init_local_model(self) -> None:
        _tok = hub_auth_token()
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, token=_tok)
        try:
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_mem
                if vram < 16 * 1024**3:
                    from transformers import BitsAndBytesConfig

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.MODEL_ID,
                        token=_tok,
                        quantization_config=bnb_config,
                        device_map="auto",
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.MODEL_ID,
                        token=_tok,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    token=_tok,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                token=_tok,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
        self.model.eval()

    def _client(self):
        if self._remote_client is None:
            from openai import OpenAI

            self._remote_client = OpenAI(
                base_url=self._remote_base_url,
                api_key=self._remote_api_key,
                timeout=self._remote_timeout,
            )
        return self._remote_client

    def generate(
        self, prompt: str, thinking: bool = True, max_new_tokens: int = 1024
    ) -> dict:
        """Run a single inference pass (remote preferred, else local).

        Parameters
        ----------
        prompt : str
            The user-facing prompt (system message is added automatically).
        thinking : bool
            When True the system message asks the model to reason inside
            ``<think>...</think>`` tags; the response is then split accordingly.
        max_new_tokens : int
            Generation budget.

        Returns
        -------
        dict
            thinking_trace : str — content of ``<think>...</think>`` (empty string
                if *thinking* is False or no tags present).
            response : str — text after ``</think>`` (or full output when no tags).
            full_output : str — complete raw model output.
        """
        system_msg = (
            "You are an expert compliance auditor. Think step by step through the legal "
            "requirements before answering. Show your reasoning in <think>...</think> tags."
            if thinking
            else "You are an expert compliance auditor. Answer concisely and directly."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        if self._use_remote:
            resp = self._client().chat.completions.create(
                model=self._remote_model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            full_output = (resp.choices[0].message.content or "").strip()
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )

            full_output = self.tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

        # Extract thinking trace and clean response
        thinking_trace = ""
        response = full_output
        if "<think>" in full_output and "</think>" in full_output:
            thinking_trace = (
                full_output.split("<think>")[1].split("</think>")[0].strip()
            )
            response = full_output.split("</think>")[-1].strip()

        return {
            "thinking_trace": thinking_trace,
            "response": response,
            "full_output": full_output,
        }


# Module-level singleton
qwen = QwenRunner()
