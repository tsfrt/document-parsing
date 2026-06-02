"""Nougat-base (facebook/nougat-base) pyfunc wrapper.

Meta AI's Nougat is a 350M-parameter VisionEncoderDecoder model purpose-built
for academic-paper / scientific-PDF parsing. It outputs Markdown-with-LaTeX
preserving math, tables, and reading order. Released August 2023 under CC-BY-NC.

Nougat is *not* a chat-style VLM -- it has no instruction following and the
``prompt`` field is intentionally ignored. Each page is processed independently
through ``processor(image)`` -> ``model.generate(pixel_values, ...)``.

Best for: technical PDFs, papers, textbooks. Less effective on photos of
receipts or scene text -- pair with Phi-3.5-vision or Granite-Vision for those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import OcrPyfunc

if TYPE_CHECKING:
    from PIL.Image import Image


class NougatPyfunc(OcrPyfunc):
    HF_REPO = "facebook/nougat-base"
    MODEL_NAME = "nougat"
    DEFAULT_PROMPT = ""  # Nougat ignores prompts; kept for interface parity.
    # Nougat-base has 4096 max position embeddings; leave headroom for the
    # encoder context tokens.
    MAX_NEW_TOKENS = 3584

    def _load(self, weights_dir: str) -> None:
        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel

        self.processor = NougatProcessor.from_pretrained(weights_dir)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(
                weights_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )

    # Nougat's image processor crashes with "axes don't match array" on
    # very small inputs (e.g. < 32 px in either dim) because the resize +
    # crop ratio math collapses to a singular tensor. Real-world PDFs and
    # photographs are always larger than this, but synthetic test images
    # and very small thumbnails can trigger it. Upscale anything smaller
    # than this threshold before feeding the processor.
    MIN_INPUT_DIM: int = 96

    def _infer_pages(self, pages: list[Image], prompt: str) -> str:
        # `prompt` is intentionally unused; Nougat is not instruction-tuned.
        del prompt
        import torch
        from PIL import Image as PILImage

        outputs: list[str] = []
        for page in pages:
            if min(page.size) < self.MIN_INPUT_DIM:
                scale = self.MIN_INPUT_DIM / min(page.size)
                new_size = (max(1, int(page.size[0] * scale)),
                            max(1, int(page.size[1] * scale)))
                page = page.resize(new_size, PILImage.LANCZOS)
            pixel_values = self.processor(images=page, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)
            with torch.inference_mode():
                generated = self.model.generate(
                    pixel_values,
                    min_length=1,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                )
            sequence = self.processor.batch_decode(
                generated, skip_special_tokens=False
            )[0]
            sequence = self.processor.post_process_generation(
                sequence, fix_markdown=False
            )
            outputs.append(sequence.strip())
        return self.join_pages(outputs)
