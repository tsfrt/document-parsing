"""Log all four OCR pyfunc models to MLflow + Unity Catalog.

Designed to be run as a Databricks Job task on serverless compute (CPU; the
download is I/O bound, no GPU needed). Snapshots HF weights once and bakes
them in as MLflow artifacts so the served container doesn't need internet
at startup.

Usage (notebook / job task):
    python -m doc_parser.log_models \
        --catalog main --schema doc_parsing \
        --models florence,phi3_vision,granite_vision,nougat
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass

import mlflow
import pandas as pd
from huggingface_hub import snapshot_download
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema

from .base import OcrPyfunc
from .models.florence import FlorencePyfunc
from .models.granite_vision import GraniteVisionPyfunc
from .models.nougat import NougatPyfunc
from .models.phi3_vision import Phi3VisionPyfunc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("log_models")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    cls: type[OcrPyfunc]
    factory: str  # filename under doc_parser/factories/ used as `python_model=<path>`
    extra_pip: tuple[str, ...] = ()


REGISTRY: dict[str, ModelSpec] = {
    # Florence-2 large-ft (Microsoft, 770 M). Replaces AllenAI's olmOCR-7B
    # which exceeded the Databricks Model Serving 300 s cold-start budget on
    # GPU_SMALL, and replaces google/paligemma2-3b which is gated. Florence-2
    # is MIT-licensed, non-gated, ~3 GB fp16, and boots in well under a
    # minute -- the smallest and fastest model in the lineup.
    "florence": ModelSpec(
        key="florence",
        cls=FlorencePyfunc,
        factory="florence_factory.py",
    ),
    "phi3_vision": ModelSpec(
        key="phi3_vision",
        cls=Phi3VisionPyfunc,
        factory="phi3_vision_factory.py",
    ),
    "granite_vision": ModelSpec(
        key="granite_vision",
        cls=GraniteVisionPyfunc,
        factory="granite_vision_factory.py",
    ),
    "nougat": ModelSpec(
        key="nougat",
        cls=NougatPyfunc,
        factory="nougat_factory.py",
    ),
}


# Pin rationale (verified against the GPU_SMALL serving container):
#   * Driver advertises CUDA 12.4 (`cudaDriverGetVersion() == 12040`). PyPI's
#     default `torch>=2.6` wheel is linked against CUDA 13 and is rejected by
#     this driver, falling the model back to CPU. torch 2.5.1's default wheel
#     is cu124-compatible.
#   * transformers must stay on 4.57.x: <4.57 lacks the processor classes we
#     trust_remote_code into; >=5.0 broke ``num_crops`` for Phi-3.5-vision
#     and the chat-template signature Granite-Vision uses.
#   * Phi-3.5-vision and Granite-Vision are still trust_remote_code repos so
#     `transformers` does not yet vendor them; we keep the dependency tight
#     to avoid silent classloader regressions.
CORE_PIP_REQUIREMENTS: tuple[str, ...] = (
    "mlflow>=2.20.0",
    "torch==2.5.1",
    "transformers~=4.57.0",
    "accelerate>=1.0.0,<2.0.0",
    "huggingface_hub>=0.26.0,<1.0.0",
    "pillow>=10.4.0",
    "pypdfium2>=4.30.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0,<2.3.0",
    "einops>=0.8.0",
    "sentencepiece>=0.2.0",
    "tiktoken>=0.8.0",
    "timm>=1.0.0",
)


def _example_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_b64": "<base64 PDF or PNG/JPEG bytes>",
                "prompt": "Convert to Markdown.",
                "output_format": "markdown",
                "max_pages": 5,
            }
        ]
    )


def _example_output() -> pd.Series:
    return pd.Series(["# Sample heading\n\nSample body."], name="output")


def _signature() -> ModelSignature:
    """`image_b64` is required; everything else has a default in OcrPyfunc."""
    inputs = Schema(
        [
            ColSpec(DataType.string, "image_b64", required=True),
            ColSpec(DataType.string, "prompt", required=False),
            ColSpec(DataType.string, "output_format", required=False),
            ColSpec(DataType.long, "max_pages", required=False),
        ]
    )
    outputs = Schema([ColSpec(DataType.string, "output", required=True)])
    return ModelSignature(inputs=inputs, outputs=outputs)


def _snapshot_weights(spec: ModelSpec, cache_dir: str) -> str:
    log.info("Downloading %s -> %s", spec.cls.HF_REPO, cache_dir)
    # Some HF repos (e.g. google/paligemma2-*) are gated and require an
    # authenticated download. ``snapshot_download`` honors HF_TOKEN /
    # HUGGING_FACE_HUB_TOKEN automatically; we pass it explicitly so the
    # log fails with a clear message ("repo gated; set HF_TOKEN secret")
    # instead of a silent 401.
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    return snapshot_download(
        repo_id=spec.cls.HF_REPO,
        revision=spec.cls.HF_REVISION,
        cache_dir=cache_dir,
        local_dir=os.path.join(cache_dir, spec.key),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.gguf", "*.onnx", "*.msgpack", "tf_*", "flax_*"],
        token=token,
    )


def _set_production_alias(registered_name: str, version: str | int) -> None:
    """Set the 'Production' alias on the freshly-logged version.

    CI uses ``databricks model-versions get-by-alias <name> Production`` to
    resolve the version number that ``databricks bundle deploy`` should target.
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(registry_uri="databricks-uc")
        client.set_registered_model_alias(
            name=registered_name,
            alias="Production",
            version=str(version),
        )
        log.info("  set alias Production -> %s/%s", registered_name, version)
    except Exception:
        log.exception("Failed to set Production alias on %s v%s", registered_name, version)


def _doc_parser_pkg_dir() -> str:
    """Return the absolute path to the `src/doc_parser` package directory.

    Used as `code_paths` so the serving container can import `doc_parser` at
    model load time. Otherwise the cloudpickle-serialized pyfunc instance
    fails to import the module that defines it.
    """
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_logged_version(registered_name: str, run_id: str) -> str:
    """Find the version number that was just registered for ``run_id``.

    `mlflow.pyfunc.log_model().registered_model_version` is sometimes None
    even on success, so fall back to searching by run id.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(registry_uri="databricks-uc")
    versions = client.search_model_versions(
        f"name='{registered_name}' AND run_id='{run_id}'"
    )
    if not versions:
        return "1"
    return str(max(int(v.version) for v in versions))


def _verify_factory_present(model_uri: str, original_factory_path: str) -> None:
    """Download the factory file from the artifact and assert it is non-empty
    and matches the size of the source we wrote. This is a guard against the
    MLflow-on-Serverless bug where shipped source can come back zero-byte.
    """
    from mlflow.artifacts import download_artifacts

    fname = os.path.basename(original_factory_path)
    src_size = os.path.getsize(original_factory_path)
    try:
        local = download_artifacts(artifact_uri=f"{model_uri}/{fname}")
    except Exception as exc:
        log.warning("Could not download %s to verify (%s)", fname, exc)
        return
    actual_size = os.path.getsize(local)
    if actual_size == 0 or actual_size < src_size // 2:
        raise RuntimeError(
            f"MLflow shipped a truncated factory: {fname} "
            f"src={src_size}B artifact={actual_size}B"
        )
    log.info("  factory verified: %s (%dB)", fname, actual_size)


_RELATIVE_IMPORT_RE = re.compile(
    r"^from \.+(?:[\w\.]*)? import .*$|^from __future__ import .*$",
    re.MULTILINE,
)


def _strip_irrelevant_lines(src: str) -> str:
    """Remove relative imports and `from __future__` lines (we'll add a single
    `from __future__ import annotations` at the top of the merged file)."""
    return _RELATIVE_IMPORT_RE.sub("", src)


def _build_self_contained_factory(spec: ModelSpec, dst_dir: str) -> str:
    """Concatenate base.py + pdf_utils.py + the model wrapper into a single
    factory file with no relative imports. Eliminates the MLflow ``code_paths``
    zero-byte-source bug we kept hitting on serverless.

    Returns the absolute path of the generated factory script.
    """
    pkg = _doc_parser_pkg_dir()
    pdf_utils_src = open(os.path.join(pkg, "pdf_utils.py")).read()
    base_src = open(os.path.join(pkg, "base.py")).read()
    model_module = spec.cls.__module__.rsplit(".", 1)[-1]  # e.g. "glm_ocr"
    model_src = open(os.path.join(pkg, "models", f"{model_module}.py")).read()

    parts = [
        '"""Self-contained MLflow factory for ' + spec.cls.__name__ + '."""',
        "from __future__ import annotations",
        "",
        "import mlflow",
        "import mlflow.models",
        "",
        "# ============================== pdf_utils ==============================",
        _strip_irrelevant_lines(pdf_utils_src),
        "",
        "# ================================ base =================================",
        _strip_irrelevant_lines(base_src),
        "",
        "# ============================== model wrapper ==========================",
        _strip_irrelevant_lines(model_src),
        "",
        "# ============================== set_model =============================",
        f"mlflow.models.set_model({spec.cls.__name__}())",
        "",
    ]
    body = "\n".join(parts)

    dst = os.path.join(dst_dir, f"{spec.key}_self_contained.py")
    with open(dst, "w") as f:
        f.write(body)
    # quick syntax check so we fail fast if the merge produced invalid python
    compile(body, dst, "exec")
    return dst


def log_one(spec: ModelSpec, *, catalog: str, schema: str, cache_dir: str) -> str:
    weights_dir = _snapshot_weights(spec, cache_dir)
    pip_reqs = list(CORE_PIP_REQUIREMENTS) + list(spec.extra_pip)
    registered_name = f"{catalog}.{schema}.{spec.key}"

    staging_root = tempfile.mkdtemp(prefix=f"docparser_pkg_{spec.key}_")
    factory_path = _build_self_contained_factory(spec, staging_root)
    log.info("Logging %s as %s (self-contained factory at %s)",
             spec.cls.__name__, registered_name, factory_path)

    with mlflow.start_run(run_name=f"log-{spec.key}") as run:
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=factory_path,
            artifacts={"weights": weights_dir},
            # No `code_paths` -- the factory file inlines base.py + pdf_utils.py
            # + the wrapper. MLflow's `code_paths` copy was producing zero-byte
            # source files for sibling modules on Serverless, which broke
            # serving-time imports.
            infer_code_paths=False,
            pip_requirements=pip_reqs,
            signature=_signature(),
            input_example=_example_input(),
            registered_model_name=registered_name,
            metadata={
                "hf_repo": spec.cls.HF_REPO,
                "model_name": spec.cls.MODEL_NAME,
            },
        )
        # Defensive verification: ensure the factory file shipped non-empty.
        _verify_factory_present(info.model_uri, factory_path)
        log.info("  run_id=%s  model_uri=%s", run.info.run_id, info.model_uri)
        version = (
            getattr(info, "registered_model_version", None)
            or _resolve_logged_version(registered_name, run.info.run_id)
        )
        _set_production_alias(registered_name, version)
    return registered_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--models",
        default=",".join(REGISTRY),
        help=f"Comma-separated list. Choices: {', '.join(REGISTRY)}",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for HF snapshots (defaults to a temp dir).",
    )
    args = parser.parse_args(argv)

    mlflow.set_registry_uri("databricks-uc")

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        parser.error(f"Unknown model(s): {unknown}. Choices: {list(REGISTRY)}")

    cache_ctx = (
        tempfile.TemporaryDirectory(prefix="hf_snapshots_")
        if args.cache_dir is None
        else None
    )
    cache_dir = args.cache_dir or cache_ctx.name  # type: ignore[union-attr]
    try:
        for k in keys:
            log_one(REGISTRY[k], catalog=args.catalog, schema=args.schema, cache_dir=cache_dir)
    finally:
        if cache_ctx is not None:
            cache_ctx.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
