# Databricks notebook source
# MAGIC %md
# MAGIC # Smoke-test all currently-READY OCR endpoints
# MAGIC
# MAGIC Sends a synthetic PNG to every `doc-parser-*` endpoint that is in
# MAGIC `state.ready == "READY"` and reports per-endpoint output + latency.

# COMMAND ----------

import base64
import io
import json
import time

from PIL import Image, ImageDraw, ImageFont
from databricks.sdk import WorkspaceClient


CANDIDATE_ENDPOINTS = [
    "doc-parser-florence",
    "doc-parser-phi3-vision",
    "doc-parser-granite-vision",
    "doc-parser-nougat",
]


def make_test_png() -> str:
    img = Image.new("RGB", (720, 280), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30
        )
    except Exception:
        font = ImageFont.load_default()
    d.text((20, 20), "Quarterly Revenue Report", fill="black", font=font)
    d.text((20, 80), "Q1 2026: $12.4M (+18% YoY)", fill="black", font=font)
    d.text((20, 130), "Q2 2026: $15.1M (+22% YoY)", fill="black", font=font)
    d.text((20, 180), "Q3 2026: $17.8M (+25% YoY)", fill="black", font=font)
    d.text((20, 230), "Total YTD: $45.3M", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# COMMAND ----------

w = WorkspaceClient()
b64 = make_test_png()

# Find which candidate endpoints are actually up.
endpoints = []
for ep_summary in w.serving_endpoints.list():
    if ep_summary.name in CANDIDATE_ENDPOINTS:
        full = w.serving_endpoints.get(ep_summary.name)
        ready = str(getattr(full.state, "ready", "")) == "EndpointStateReady.READY"
        endpoints.append({"name": full.name, "ready": ready})
print(json.dumps(endpoints, indent=2))

# COMMAND ----------

results: list[dict] = []
for entry in endpoints:
    name = entry["name"]
    if not entry["ready"]:
        results.append({"endpoint": name, "ok": False, "skipped": "not READY"})
        continue
    try:
        t0 = time.time()
        resp = w.serving_endpoints.query(
            name=name,
            dataframe_records=[{"image_b64": b64}],
        )
        elapsed = time.time() - t0
        preds = resp.predictions if hasattr(resp, "predictions") else resp
        text = ""
        if preds:
            first = preds[0]
            text = first.get("output", "") if isinstance(first, dict) else str(first)
        results.append({
            "endpoint": name,
            "ok": bool(text),
            "elapsed_s": round(elapsed, 2),
            "len_chars": len(text or ""),
            "preview": (text or "")[:400],
        })
    except Exception as exc:
        results.append({
            "endpoint": name,
            "ok": False,
            "error": str(exc)[:600],
        })

print(json.dumps(results, indent=2, default=str))
display(spark.createDataFrame(results))  # type: ignore[name-defined]  # noqa: F821
dbutils.notebook.exit(json.dumps(results, default=str)[:60000])
