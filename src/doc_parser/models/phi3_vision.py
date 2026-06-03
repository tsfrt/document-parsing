"""Phi-3.5-vision-instruct (microsoft/Phi-3.5-vision-instruct) pyfunc wrapper.

A 4.2B-parameter Microsoft multimodal model with strong document understanding
and OCR capabilities. Released April 2024 under MIT.

bf16 weights are ~8.4 GB; with image-encoder activations and KV cache the
peak working set on 4 K-token outputs is ~10-11 GB which fits cleanly on a
16 GB T4 without quantization.

T4 lacks Flash-Attention-2 support so we explicitly request the eager attention
implementation; the loader otherwise tries to import flash_attn and crashes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import OcrPyfunc

if TYPE_CHECKING:
    from PIL.Image import Image


class Phi3VisionPyfunc(OcrPyfunc):
    HF_REPO = "microsoft/Phi-3.5-vision-instruct"
    MODEL_NAME = "phi3_vision"
    DEFAULT_PROMPT = (
        "Extract all visible text from this document page as clean Markdown. "
        "Preserve headings, lists, and tables as Markdown tables. "
        "Render equations as LaTeX inside $...$ or $$...$$."
    )
    MAX_NEW_TOKENS = 4096

    def _load(self, weights_dir: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Phi-3.5-vision's `modeling_phi3_v.py` uses ``DynamicCache.seen_tokens``,
        # which transformers >= 4.50 removed (the public surface is now
        # ``DynamicCache.get_seq_length()``). Restore the attribute as a
        # property *before* the model is loaded -- the cache is constructed
        # lazily on the first generate() call.
        self._patch_dynamic_cache_seen_tokens()

        # `num_crops=4` is Phi-3.5-vision's recommended setting for a single
        # high-resolution image (single-frame document mode).
        self.processor = AutoProcessor.from_pretrained(
            weights_dir, trust_remote_code=True, num_crops=4
        )
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                weights_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                _attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _patch_dynamic_cache_seen_tokens() -> None:
        """Restore the legacy ``DynamicCache`` API surface that Phi-3.5-vision's
        vendored modeling code relies on but transformers >= 4.50 removed:

        * ``DynamicCache.seen_tokens``        -> ``get_seq_length()``
        * ``DynamicCache.get_max_length()``   -> ``get_max_cache_shape()``
        * ``DynamicCache.get_usable_length()``-> ``get_seq_length(layer_idx)``
          (DynamicCache has no max length, so usable == current.)
        """
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            return

        if not hasattr(DynamicCache, "seen_tokens"):
            def _seen_tokens(self):  # noqa: ANN001 - matches transformers cache surface
                try:
                    return self.get_seq_length()
                except Exception:
                    return 0

            DynamicCache.seen_tokens = property(_seen_tokens)

        if not hasattr(DynamicCache, "get_max_length"):
            def _get_max_length(self):  # noqa: ANN001
                fn = getattr(self, "get_max_cache_shape", None)
                if fn is not None:
                    try:
                        return fn()
                    except Exception:
                        return None
                return None

            DynamicCache.get_max_length = _get_max_length

        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length, layer_idx=0):  # noqa: ANN001
                # DynamicCache has no max length -- usable == seq length so far.
                try:
                    return self.get_seq_length(layer_idx)
                except Exception:
                    return 0

            DynamicCache.get_usable_length = _get_usable_length

    def _infer_pages(self, pages: list[Image], prompt: str) -> str:
        import torch

        outputs: list[str] = []
        for page in pages:
            # Phi-3.5-vision requires the inline `<|image_1|>` placeholder in
            # the user message; the processor maps it to the image tensor.
            messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
            chat_text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(chat_text, [page], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated[:, input_len:]
            text = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            outputs.append(text.strip())
        return self.join_pages(outputs)
