"""Granite-Vision-3.2-2B (ibm-granite/granite-vision-3.2-2b) pyfunc wrapper.

IBM Research's compact (2.5 B params) vision-language model fine-tuned for
document understanding: tables, charts, infographics, and forms. Built on a
LLaVA-style architecture with the Granite-3.0-2B language backbone. Released
February 2025 under Apache 2.0.

bf16 weights are ~5 GB; comfortably fits in 16 GB VRAM with room for batching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import OcrPyfunc

if TYPE_CHECKING:
    from PIL.Image import Image


class GraniteVisionPyfunc(OcrPyfunc):
    HF_REPO = "ibm-granite/granite-vision-3.2-2b"
    MODEL_NAME = "granite_vision"
    DEFAULT_PROMPT = (
        "Convert this document page to clean Markdown. "
        "Preserve headings, lists, tables (as Markdown tables), and the "
        "natural reading order. Do not hallucinate."
    )
    MAX_NEW_TOKENS = 4096

    def _load(self, weights_dir: str) -> None:
        import torch
        import transformers

        self.processor = transformers.AutoProcessor.from_pretrained(
            weights_dir, trust_remote_code=True
        )
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Granite-Vision 3.2 is registered as LlavaNextForConditionalGeneration.
        # AutoModelForImageTextToText resolves it correctly on transformers
        # >= 4.57 -- fall back to AutoModelForVision2Seq for older builds.
        last_err: Exception | None = None
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                self.model = (
                    cls.from_pretrained(
                        weights_dir,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                    )
                    .to(self.device)
                    .eval()
                )
                return
            except (ValueError, KeyError) as exc:
                if "Unrecognized configuration" in str(exc):
                    last_err = exc
                    continue
                raise
        raise RuntimeError(f"Could not load Granite-Vision: {last_err}")

    def _infer_pages(self, pages: list[Image], prompt: str) -> str:
        import torch

        outputs: list[str] = []
        for page in pages:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": page},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated[:, input_len:]
            text = self.processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0]
            outputs.append(text.strip())
        return self.join_pages(outputs)
