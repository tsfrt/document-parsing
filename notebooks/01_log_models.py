# Databricks notebook source
# MAGIC %md
# MAGIC # Log all four OCR pyfunc models to Unity Catalog
# MAGIC
# MAGIC Snapshots each Hugging Face repo locally then logs the model with
# MAGIC weights baked in as MLflow artifacts so the served container does not
# MAGIC need internet at startup. Runs on serverless compute (CPU only).

# COMMAND ----------

# MAGIC %pip install -q -U mlflow>=2.20.0 huggingface_hub>=0.26.0 pypdfium2>=4.30.0 pillow>=10.4.0
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "doc_parsing")
dbutils.widgets.text("models", "florence,phi3_vision,granite_vision,nougat")

# COMMAND ----------

import os
import sys


def _discover_src_dir() -> str:
    """Find the bundled `src/` directory.

    When `databricks bundle deploy` uploads this notebook, the src/ tree is
    placed alongside under `<bundle_root>/files/src/`. We walk up from the
    notebook path and pick the first ancestor that contains a `src/`.
    """
    nb_path = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # type: ignore[name-defined]  # noqa: F821
    )
    cur = "/Workspace" + os.path.dirname(nb_path)
    for _ in range(8):
        candidate = os.path.join(cur, "src")
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "doc_parser")):
            return candidate
        cur = os.path.dirname(cur)
    raise RuntimeError("Could not locate src/doc_parser relative to this notebook.")


SRC = _discover_src_dir()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
print(f"Using src: {SRC}")

# COMMAND ----------

from doc_parser.log_models import main as log_models_main

argv = [
    "--catalog", dbutils.widgets.get("catalog"),
    "--schema", dbutils.widgets.get("schema"),
    "--models", dbutils.widgets.get("models"),
]
log_models_main(argv)
