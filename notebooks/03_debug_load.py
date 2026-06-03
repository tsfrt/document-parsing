# Databricks notebook source
# MAGIC %md
# MAGIC # Debug processor / tokenizer loading for the three OCR models
# MAGIC
# MAGIC Runs on serverless CPU. For each HF repo we snapshot the weights once
# MAGIC then try several class-resolution strategies (`AutoProcessor`,
# MAGIC `AutoTokenizer`, `AutoImageProcessor`, plus the `auto_map` advertised by
# MAGIC the repo). Findings are returned via `dbutils.notebook.exit()` and also
# MAGIC written to /Volumes/<catalog>/<schema>/sample_docs/debug_load.json.

# COMMAND ----------

# MAGIC %pip install -q -U \
# MAGIC   "transformers>=4.57.0" \
# MAGIC   "huggingface_hub>=0.26.0" \
# MAGIC   "pillow>=10.4.0" \
# MAGIC   "einops>=0.8.0" \
# MAGIC   "sentencepiece>=0.2.0" \
# MAGIC   "tiktoken>=0.8.0" \
# MAGIC   "timm>=1.0.0"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "serverless_stable_zkm2ao_catalog")
dbutils.widgets.text("schema", "doc_parsing")

# COMMAND ----------

import importlib.util
import json
import os
import sys
import tempfile
import traceback

from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)

REPOS = {
    "florence":       "microsoft/Florence-2-large-ft",
    "phi3_vision":    "microsoft/Phi-3.5-vision-instruct",
    "granite_vision": "ibm-granite/granite-vision-3.2-2b",
}

results: dict[str, dict] = {}


def attempt(label: str, fn) -> dict:
    try:
        out = fn()
        return {"ok": True, "result": type(out).__name__}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:400],
        }


def _load_remote_class(local: str, dotted: str):
    file_name, cls_name = dotted.split(".")
    file_path = os.path.join(local, f"{file_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    spec = importlib.util.spec_from_file_location(file_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[file_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


cache_root = tempfile.mkdtemp(prefix="hf_debug_")

# COMMAND ----------

for key, repo in REPOS.items():
    entry: dict = {"repo": repo}
    try:
        local = snapshot_download(
            repo_id=repo,
            cache_dir=cache_root,
            local_dir=os.path.join(cache_root, key),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.gguf", "*.onnx", "*.msgpack", "tf_*", "flax_*", "*.bin"],
        )
        entry["snapshot"] = local
        entry["files"] = sorted(os.listdir(local))
    except Exception as exc:
        entry["snapshot_error"] = f"{type(exc).__name__}: {exc}"
        results[key] = entry
        continue

    # --- inspect configs
    cfg = {}
    for fname in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        p = os.path.join(local, fname)
        if not os.path.exists(p):
            continue
        try:
            cfg[fname] = json.load(open(p))
        except Exception as exc:
            cfg[fname] = f"<<unreadable: {exc}>>"
    entry["configs"] = {
        fname: {
            k: data.get(k)
            for k in (
                "model_type",
                "architectures",
                "auto_map",
                "tokenizer_class",
                "image_processor_type",
                "processor_class",
            )
            if isinstance(data, dict) and data.get(k) is not None
        }
        for fname, data in cfg.items()
        if isinstance(data, dict)
    }

    # --- try every reasonable class-loading strategy
    entry["probes"] = {
        "AutoConfig + trust_remote_code":
            attempt("AutoConfig", lambda: AutoConfig.from_pretrained(local, trust_remote_code=True)),
        "AutoTokenizer + trust_remote_code":
            attempt("AutoTokenizer", lambda: AutoTokenizer.from_pretrained(local, trust_remote_code=True)),
        "AutoTokenizer + trust_remote_code, use_fast=False":
            attempt("AutoTokenizerSlow", lambda: AutoTokenizer.from_pretrained(local, trust_remote_code=True, use_fast=False)),
        "AutoImageProcessor + trust_remote_code":
            attempt("AutoImageProcessor", lambda: AutoImageProcessor.from_pretrained(local, trust_remote_code=True)),
        "AutoProcessor + trust_remote_code":
            attempt("AutoProcessor", lambda: AutoProcessor.from_pretrained(local, trust_remote_code=True)),
        "AutoProcessor + trust_remote_code, use_fast=True":
            attempt("AutoProcessorFast", lambda: AutoProcessor.from_pretrained(local, trust_remote_code=True, use_fast=True)),
    }

    # --- try architecture-specific classes via auto_map
    candidates: dict[str, str] = {}
    for fname, data in cfg.items():
        if not isinstance(data, dict):
            continue
        am = data.get("auto_map") or {}
        for hf_key, dotted in am.items():
            if isinstance(dotted, list):
                dotted = dotted[0]
            if isinstance(dotted, str) and "." in dotted:
                candidates[hf_key] = dotted
    auto_map_results = {}
    for hf_key, dotted in candidates.items():
        try:
            cls = _load_remote_class(local, dotted)
            auto_map_results[hf_key] = {"ok": True, "dotted": dotted, "class": cls.__name__}
        except Exception as exc:
            auto_map_results[hf_key] = {
                "ok": False,
                "dotted": dotted,
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:400],
            }
    entry["auto_map"] = auto_map_results

    results[key] = entry

# COMMAND ----------

# Persist a copy to UC volume so we can read it back via the SDK + return via notebook.exit
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
out_path = f"/Volumes/{catalog}/{schema}/sample_docs/debug_load.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"wrote {out_path}")

dbutils.notebook.exit(json.dumps(results, default=str)[:65000])
