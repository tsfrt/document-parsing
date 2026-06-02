# Databricks notebook source
# MAGIC %md
# MAGIC # Smoke test the OCR endpoints
# MAGIC
# MAGIC Sends a tiny synthetic PNG to each `doc-parser-*` endpoint and asserts a
# MAGIC non-empty markdown string is returned. Runs after every deploy via the
# MAGIC `smoke-test-ocr-endpoints` job. Failures here fail the deploy.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "doc_parsing")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

import base64
import io
import json
import time

from PIL import Image, ImageDraw, ImageFont
from databricks.sdk import WorkspaceClient

ENDPOINTS = [
    "doc-parser-florence",
    "doc-parser-phi3-vision",
    "doc-parser-granite-vision",
    "doc-parser-nougat",
]


def make_test_png() -> str:
    img = Image.new("RGB", (640, 200), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 60), "Hello Databricks OCR", fill="black", font=font)
    draw.text((20, 120), "1234567890", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_endpoint(client: WorkspaceClient, name: str, image_b64: str):
    t0 = time.time()
    resp = client.serving_endpoints.query(
        name=name,
        dataframe_records=[{"image_b64": image_b64}],
    )
    elapsed = time.time() - t0
    return resp, elapsed


# COMMAND ----------

w = WorkspaceClient()
image_b64 = make_test_png()

results: list[dict] = []
for ep in ENDPOINTS:
    try:
        resp, elapsed = call_endpoint(w, ep, image_b64)
        preds = resp.predictions if hasattr(resp, "predictions") else resp
        text = preds[0] if preds else ""
        ok = bool(text and isinstance(text, str))
        results.append(
            {"endpoint": ep, "ok": ok, "elapsed_s": round(elapsed, 2), "preview": (text or "")[:120]}
        )
    except Exception as exc:
        results.append({"endpoint": ep, "ok": False, "elapsed_s": None, "error": str(exc)})

display(spark.createDataFrame(results))  # type: ignore[name-defined]  # noqa: F821

failed = [r for r in results if not r.get("ok")]
if failed:
    raise AssertionError(f"Smoke test failed for: {json.dumps(failed, indent=2)}")
print("All endpoints OK.")
