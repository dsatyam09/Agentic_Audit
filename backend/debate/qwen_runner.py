"""QwenRunner — singleton wrapper around a Qwen model for compliance debate inference.

Handles VRAM detection, 4-bit quantization fallback, and structured output
with thinking-trace extraction.

The model ID is read from the QWEN_MODEL_ID environment variable, defaulting
to Qwen/Qwen3-8B.  For testing on CPU without downloading 8B weights, set:
    QWEN_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenRunner:
    MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-8B")

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        # Check VRAM: if < 16GB, use 4-bit quantization
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
                # CPU fallback — use float32
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

    def generate(
        self, prompt: str, thinking: bool = True, max_new_tokens: int = 1024
    ) -> dict:
        """Run a single inference pass on the loaded Qwen3-8B model.

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
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        full_output = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
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
