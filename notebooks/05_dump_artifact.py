# Databricks notebook source
# MAGIC %md
# MAGIC # Dump the contents of one or more model artifacts
# MAGIC
# MAGIC Used during debug to verify which versions ship a complete `code/` tree.
# MAGIC Pass a comma-separated list of `name@alias` or `name#version` specs.

# COMMAND ----------

dbutils.widgets.text(
    "specs",
    "serverless_stable_zkm2ao_catalog.doc_parsing.florence@Production",
)

# COMMAND ----------

import json
import os

import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

specs = [s.strip() for s in dbutils.widgets.get("specs").split(",") if s.strip()]
out: dict = {}

for spec in specs:
    if "@" in spec:
        name, alias = spec.split("@", 1)
        mv = client.get_model_version_by_alias(name, alias)
        version = mv.version
    elif "#" in spec:
        name, version = spec.split("#", 1)
    else:
        name, version = spec, None

    uri = (
        f"models:/{name}/{version}"
        if version is not None
        else f"models:/{name}@Production"
    )
    code_uri = f"{uri}/code"
    try:
        local = download_artifacts(artifact_uri=code_uri)
    except Exception as exc:
        out[spec] = {"version": version, "error": str(exc)[:300]}
        continue

    files: dict[str, dict] = {}
    for root, _, fs in os.walk(local):
        for f in fs:
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, local)
            sz = os.path.getsize(full)
            txt = ""
            if sz < 200_000:
                txt = open(full).read()
            files[rel] = {
                "size": sz,
                "head": txt[:200].replace("\n", "\\n"),
            }
    out[spec] = {"version": version, "files": files}

print(json.dumps(out, indent=2)[:50000])
dbutils.notebook.exit(json.dumps(out, default=str)[:60000])
