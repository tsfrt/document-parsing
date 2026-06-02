"""Shared base class for the OCR pyfunc wrappers.

The base handles the boilerplate that is identical across every model:
  * input parsing (DataFrame -> per-row dicts)
  * base64 decoding (PDF or raster image)
  * page-level inference loop with optional max_pages clamp
  * markdown vs JSON output selection
  * minimal error handling so a malformed row doesn't poison the batch

Subclasses implement only the model-specific bits in `_load` and `_infer_pages`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import mlflow
import pandas as pd

from .pdf_utils import DEFAULT_PDF_DPI, decode_pages

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


PAGE_SEPARATOR = "\n\n---\n\n"


class OcrPyfunc(mlflow.pyfunc.PythonModel):
    """Base class for HF-backed OCR pyfunc models.

    Subclasses must override:
      * HF_REPO       - the Hugging Face repository id (e.g. ``zai-org/GLM-OCR``).
      * MODEL_NAME    - short, lowercase identifier ("florence", "phi3_vision", ...).

    Subclasses should override:
      * DEFAULT_PROMPT
      * _load(weights_dir)        - load processor/model, set torch_dtype/device.
      * _infer_pages(pages, ...)  - run inference on one document's pages,
                                    return a single markdown string.
    """

    HF_REPO: str = ""
    HF_REVISION: str | None = None
    MODEL_NAME: str = ""
    DEFAULT_PROMPT: str = "Extract all text from this document as Markdown. Preserve tables as Markdown tables."
    DEFAULT_DPI: int = DEFAULT_PDF_DPI
    MAX_PAGES_PER_DOC: int = 50

    # Set in load_context.
    processor: Any = None
    model: Any = None
    device: str = "cuda"

    # ------------------------------------------------------------------ MLflow

    def load_context(self, context):  # noqa: D401 - MLflow API
        weights_dir = context.artifacts["weights"]
        log.info("Loading %s from %s", self.HF_REPO, weights_dir)
        self._select_device()
        self._load(weights_dir)

    def predict(self, context, model_input, params=None):  # noqa: D401 - MLflow API
        rows = self._coerce_input(model_input)
        results: list[str] = []
        for row in rows:
            try:
                results.append(self._handle_row(row))
            except Exception as exc:  # surface error per-row
                log.exception("OCR inference failed for row")
                results.append(json.dumps({"error": str(exc)}))
        return pd.Series(results, name="output")

    # -------------------------------------------------------------- subclasses

    def _load(self, weights_dir: str) -> None:
        """Load processor + model from a local snapshot directory."""
        raise NotImplementedError

    def _load_auto_model_with_fallback(
        self,
        weights_dir: str,
        *,
        attn_implementation: str | None = None,
    ):
        """Try the modern Vision2Seq/ImageTextToText classes first then fall
        back to AutoModelForCausalLM / AutoModel. Different repos register
        under different Auto* classes (e.g. Granite-Vision -> ImageTextToText,
        Phi-3.5-vision -> CausalLM via auto_map), so we walk the candidate
        list until one accepts the config.
        """
        import torch
        import transformers

        candidates = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
            "AutoModel",
        ]
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        last_err: Exception | None = None
        for cls_name in candidates:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(weights_dir, **kwargs)
                log.info("  loaded model with %s -> %s", cls_name, type(model).__name__)
                return model
            except ValueError as exc:
                if "Unrecognized configuration class" in str(exc):
                    last_err = exc
                    continue
                raise
        raise RuntimeError(
            f"None of {candidates} accepted the config at {weights_dir}: {last_err}"
        )

    def _infer_pages(self, pages: list[Image], prompt: str) -> str:
        """Run inference over the pages of one document, return a single string."""
        raise NotImplementedError

    # ----------------------------------------------------------------- helpers

    def _select_device(self) -> None:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            self.device = "cuda" if cuda_available else "cpu"
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_capability = torch.cuda.get_device_capability(0)
                log.info(
                    "DEVICE=cuda  torch=%s  cuda_runtime=%s  gpu=%s  capability=%s",
                    torch.__version__,
                    torch.version.cuda,
                    gpu_name,
                    gpu_capability,
                )
            else:
                # When this happens you almost certainly have a torch wheel /
                # driver mismatch. Surface enough info to debug from logs.
                log.warning(
                    "DEVICE=cpu  torch=%s  cuda_runtime=%s  cuda.is_available=%s",
                    torch.__version__,
                    getattr(torch.version, "cuda", "?"),
                    cuda_available,
                )
        except Exception as exc:
            log.exception("DEVICE selection failed; falling back to cpu (%s)", exc)
            self.device = "cpu"
        # Cap intra-op threads on CPU containers to keep latency predictable.
        os.environ.setdefault("OMP_NUM_THREADS", "4")

    def _coerce_input(self, model_input) -> list[dict[str, Any]]:
        if isinstance(model_input, pd.DataFrame):
            return model_input.to_dict(orient="records")
        if isinstance(model_input, list):
            return [r if isinstance(r, dict) else {"image_b64": r} for r in model_input]
        if isinstance(model_input, dict):
            # Single record dict.
            return [model_input]
        raise ValueError(f"Unsupported input type: {type(model_input).__name__}")

    def _handle_row(self, row: dict[str, Any]) -> str:
        image_b64 = row.get("image_b64")
        if not image_b64:
            raise ValueError("Missing required field 'image_b64'")
        prompt = row.get("prompt") or self.DEFAULT_PROMPT
        output_format = (row.get("output_format") or "markdown").lower()
        max_pages = int(row.get("max_pages") or self.MAX_PAGES_PER_DOC)

        pages = decode_pages(
            image_b64,
            dpi=self.DEFAULT_DPI,
            max_pages=max_pages,
        )
        if not pages:
            raise ValueError("Document had no pages after decoding")

        markdown = self._infer_pages(pages, prompt)
        if output_format == "json":
            return json.dumps(
                {
                    "model": self.MODEL_NAME,
                    "num_pages": len(pages),
                    "text": markdown,
                },
                ensure_ascii=False,
            )
        return markdown

    @staticmethod
    def join_pages(per_page: list[str]) -> str:
        return PAGE_SEPARATOR.join(s.strip() for s in per_page if s)
