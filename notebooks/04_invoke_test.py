# Databricks notebook source
# MAGIC %md
# MAGIC # Invoke a single OCR endpoint with a synthetic image
# MAGIC
# MAGIC Used to manually validate one endpoint at a time during debug iteration.

# COMMAND ----------

dbutils.widgets.text("endpoint", "doc-parser-florence")

# COMMAND ----------

import base64
import io
import json
import time

from PIL import Image, ImageDraw, ImageFont
from databricks.sdk import WorkspaceClient


def make_test_png() -> str:
    img = Image.new("RGB", (640, 240), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
    d.text((20, 30), "Quarterly Revenue Report", fill="black", font=font)
    d.text((20, 90), "Q1 2026: $12.4M (+18% YoY)", fill="black", font=font)
    d.text((20, 140), "Q2 2026: $15.1M (+22% YoY)", fill="black", font=font)
    d.text((20, 190), "Total H1: $27.5M", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


name = dbutils.widgets.get("endpoint")
b64 = make_test_png()
w = WorkspaceClient()

t0 = time.time()
try:
    resp = w.serving_endpoints.query(
        name=name,
        dataframe_records=[{"image_b64": b64}],
    )
    elapsed = time.time() - t0
    preds = resp.predictions if hasattr(resp, "predictions") else resp
    out = {"endpoint": name, "elapsed_s": round(elapsed, 2), "preview": str(preds)[:1500]}
except Exception as exc:
    out = {"endpoint": name, "elapsed_s": round(time.time() - t0, 2), "error": str(exc)[:1500]}

print(json.dumps(out, indent=2))
dbutils.notebook.exit(json.dumps(out, default=str)[:60000])
