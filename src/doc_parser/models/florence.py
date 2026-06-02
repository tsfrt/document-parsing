"""Florence-2 large-ft (microsoft/Florence-2-large-ft) pyfunc wrapper.

Microsoft Research's compact (770 M params, ~3 GB fp16) vision-language
model with a unified prompt-based task interface. The "-ft" variant is
fine-tuned on a more diverse mix of tasks; ``<OCR>`` is the relevant one
for document text extraction. Released June 2024 under the MIT license,
non-gated on Hugging Face.

Why this lineup slot: replaces ``google/paligemma2-3b-mix-448`` to avoid
the gated-repo HF token requirement while keeping a small, fast-booting
USA-derived OCR model. Boots in well under a minute on T4.

Notes for serving:
* The official Florence-2 modeling code lives in the HF repo (custom
  ``Florence2ForConditionalGeneration``); we load with ``trust_remote_code``.
* Florence-2 is task-prompted, not chat-prompted. The wrapper supports
  the standard task tokens (``<OCR>``, ``<OCR_WITH_REGION>``,
  ``<DETAILED_CAPTION>``, ``<MORE_DETAILED_CAPTION>``, ``<CAPTION>``,
  ``<DENSE_REGION_CAPTION>``, ``<REGION_PROPOSAL>``, ``<OD>``,
  ``<CAPTION_TO_PHRASE_GROUNDING>``); free-form prompts are passed
  through as-is and Florence will attempt to parse them.
* ``processor.post_process_generation`` returns a dict keyed by the task
  token; we extract the matching value (or the raw decode for free-form
  prompts) so downstream callers always get a string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import OcrPyfunc

if TYPE_CHECKING:
    from PIL.Image import Image


class FlorencePyfunc(OcrPyfunc):
    HF_REPO = "microsoft/Florence-2-large-ft"
    MODEL_NAME = "florence"
    DEFAULT_PROMPT = "<OCR>"
    MAX_NEW_TOKENS = 1024

    # Recognised Florence-2 task tokens that the processor's
    # ``post_process_generation`` knows how to render. Any prompt outside
    # this set is treated as free-form and we fall back to the raw decode.
    _KNOWN_TASKS: tuple[str, ...] = (
        "<OCR>",
        "<OCR_WITH_REGION>",
        "<CAPTION>",
        "<DETAILED_CAPTION>",
        "<MORE_DETAILED_CAPTION>",
        "<DENSE_REGION_CAPTION>",
        "<REGION_PROPOSAL>",
        "<OD>",
        "<CAPTION_TO_PHRASE_GROUNDING>",
    )

    def _load(self, weights_dir: str) -> None:
        import torch
        import transformers

        # Florence-2 is registered as ``Florence2ForConditionalGeneration``
        # via trust_remote_code. AutoModelForCausalLM is what the model
        # card uses; AutoModelForVision2Seq is a defensive fallback in
        # case a future transformers release re-classifies it.
        self.processor = transformers.AutoProcessor.from_pretrained(
            weights_dir, trust_remote_code=True
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Florence-2's custom modeling code (modeling_florence2.py) predates
        # transformers' SDPA dispatch contract: it does not declare the
        # ``_supports_sdpa`` class attribute that ``_check_and_adjust_attn_implementation``
        # consults at __init__ time. Passing ``attn_implementation="eager"``
        # short-circuits that check so the model loads on transformers 4.57+.
        # Eager attention has no measurable impact on Florence-2 latency on
        # T4 (sequence lengths are tiny and the encoder is the hot path).
        last_err: Exception | None = None
        for cls_name in ("AutoModelForCausalLM", "AutoModelForVision2Seq"):
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
                        attn_implementation="eager",
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
        raise RuntimeError(f"Could not load Florence-2: {last_err}")

    def _infer_pages(self, pages: list[Image], prompt: str) -> str:
        import torch

        task_token = prompt if prompt.startswith("<") else None
        outputs: list[str] = []
        for page in pages:
            inputs = self.processor(
                text=prompt,
                images=page,
                return_tensors="pt",
            ).to(self.device, dtype=self.model.dtype if self.device == "cuda" else None)
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    num_beams=3,
                    do_sample=False,
                )
            decoded = self.processor.batch_decode(
                generated, skip_special_tokens=False
            )[0]
            text = self._postprocess(decoded, task_token, page.size)
            outputs.append(text.strip())
        return self.join_pages(outputs)

    def _postprocess(self, decoded: str, task_token: str | None, image_size: tuple[int, int]) -> str:
        """Run Florence's task-specific post-processor when the prompt is
        a recognised task token; fall back to the raw decode otherwise."""
        if task_token in self._KNOWN_TASKS:
            try:
                parsed = self.processor.post_process_generation(
                    decoded, task=task_token, image_size=image_size
                )
                value = parsed.get(task_token, decoded)
                # ``post_process_generation`` returns dict/list for
                # bbox-style tasks; OCR tasks return strings. Stringify
                # anything that's not already a string so the SQL UDF
                # contract is preserved.
                if isinstance(value, str):
                    return value
                return str(value)
            except Exception:
                pass
        # Free-form / unknown prompt: strip the special tokens.
        return self.processor.tokenizer.decode(
            self.processor.tokenizer(decoded, return_tensors="pt").input_ids[0],
            skip_special_tokens=True,
        )
