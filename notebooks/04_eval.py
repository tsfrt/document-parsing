# Databricks notebook source
# MAGIC %md
# MAGIC # OCR / document-parsing evaluation across the three endpoints
# MAGIC
# MAGIC Compares the three `doc-parser-*` serving endpoints on a small, curated set
# MAGIC of synthetic documents. The dataset and scorers are deliberately
# MAGIC OCR-shaped (not generic GenAI):
# MAGIC
# MAGIC ### Document mix
# MAGIC | Type             | What it stresses                                       |
# MAGIC |------------------|--------------------------------------------------------|
# MAGIC | prose paragraph  | baseline character-level accuracy on running text      |
# MAGIC | numeric receipt  | digit / currency / decimal fidelity, column alignment  |
# MAGIC | markdown table   | structure preservation (cells, header row, ordering)   |
# MAGIC | key:value form   | layout-aware extraction of fielded data                |
# MAGIC | ordered list     | reading-order preservation (1, 2, 3, ...)              |
# MAGIC | scientific PDF   | dense paragraph + a simple equation, multi-page PDF    |
# MAGIC
# MAGIC ### Scorers (OCR-focused)
# MAGIC | Scorer              | Type             | What it measures                                            |
# MAGIC |---------------------|------------------|-------------------------------------------------------------|
# MAGIC | `cer`               | deterministic    | Character Error Rate vs `expected_response` (lower better). |
# MAGIC | `char_accuracy`     | deterministic    | `1 - cer` so it aggregates upwards (higher better).         |
# MAGIC | `wer`               | deterministic    | Word Error Rate (whitespace tokens).                        |
# MAGIC | `text_similarity`   | deterministic    | `rapidfuzz.token_set_ratio / 100` -- robust to reorder.     |
# MAGIC | `numeric_fidelity`  | deterministic    | Recall over the digits/currency/dates that MUST appear.     |
# MAGIC | `reading_order`     | deterministic    | Expected lines appear in the same order in the output.      |
# MAGIC | `table_structure`   | deterministic    | For the table doc, output contains a markdown table.        |
# MAGIC | `ocr_quality`       | LLM judge        | OCR-specific rubric (chars, numbers, order, structure).     |
# MAGIC | `latency_s`         | infra            | End-to-end endpoint round-trip.                             |
# MAGIC | `non_empty`         | infra            | Endpoint returned a non-empty string.                       |
# MAGIC
# MAGIC One named MLflow run per endpoint, all in the same experiment.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1.0" "databricks-sdk>=0.40.0" "pillow>=10.4.0" "pypdfium2>=4.30.0" "rapidfuzz>=3.10.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("experiment_path", "/Shared/ai-parse-doc/eval")
dbutils.widgets.text(
    "endpoints",
    "doc-parser-florence,doc-parser-phi3-vision,doc-parser-granite-vision",
)
dbutils.widgets.text("judge_model", "databricks:/databricks-meta-llama-3-3-70b-instruct")

EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")
ENDPOINTS = [e.strip() for e in dbutils.widgets.get("endpoints").split(",") if e.strip()]
JUDGE_MODEL = dbutils.widgets.get("judge_model")

# COMMAND ----------

import base64
import io
import re
import time
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import token_set_ratio

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Logging to experiment: {EXPERIMENT_PATH}")
print(f"Endpoints under test:  {ENDPOINTS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthesise the OCR ground-truth dataset
# MAGIC
# MAGIC Each record carries:
# MAGIC   * `inputs.image_b64` -- bytes sent to the endpoint.
# MAGIC   * `inputs.description` / `inputs.doc_type` -- shows up in traces and is
# MAGIC     used by structure-aware scorers (e.g. `table_structure` only fires on
# MAGIC     `doc_type == "table"`).
# MAGIC   * `expectations.expected_response` -- canonical reference text (used by
# MAGIC     CER / WER / text similarity / `ocr_quality` judge).
# MAGIC   * `expectations.numeric_tokens` -- digits/currency/dates that must
# MAGIC     survive the OCR exactly.
# MAGIC   * `expectations.ordered_lines` -- lines that should appear in this order.

# COMMAND ----------


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _png(lines: list[str], *, width: int = 820, line_height: int = 52) -> str:
    height = 40 + line_height * len(lines) + 20
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    font = _font(28)
    y = 20
    for line in lines:
        d.text((24, y), line, fill="black", font=font)
        y += line_height
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _multipage_pdf(pages: list[list[str]]) -> str:
    imgs: list[Image.Image] = []
    for lines in pages:
        img = Image.new("RGB", (816, 1056), "white")  # ~ US Letter @ 96 dpi
        d = ImageDraw.Draw(img)
        font = _font(26)
        y = 80
        for line in lines:
            d.text((80, y), line, fill="black", font=font)
            y += 50
        imgs.append(img)
    buf = io.BytesIO()
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
    return base64.b64encode(buf.getvalue()).decode()


_RECEIPT_LINES = [
    "ACME HARDWARE  -  Receipt",
    "Date: 2026-04-15   Order #4421",
    "1x Hammer          $14.99",
    "3x Wood Screws     $ 7.50",
    "1x Drill Bit Set   $29.95",
    "Subtotal           $52.44",
    "Tax (8.25%)        $ 4.33",
    "Total              $56.77",
]

_TABLE_LINES = [
    "Region   |  Q1     |  Q2     |  Q3",
    "------------------------------------",
    "North    |  120.5  |  138.2  |  151.7",
    "South    |   88.1  |   95.4  |  102.9",
    "East     |  142.0  |  150.8  |  167.3",
    "West     |   76.6  |   81.2  |   89.4",
]

_FORM_LINES = [
    "Patient: Jane Doe          DOB: 1984-07-12",
    "MRN: 00457821              Sex: F",
    "Diagnosis: Hypertension (I10)",
    "Allergies: Penicillin",
    "Rx: Lisinopril 10mg, once daily",
    "Provider: Dr. R. Patel",
]

_LIST_LINES = [
    "How to deploy a Databricks Asset Bundle",
    "1. Run databricks bundle validate",
    "2. Run databricks bundle deploy --target dev",
    "3. Run the log-ocr-models job",
    "4. Resolve aliases and re-deploy",
    "5. Apply the SQL UDFs",
    "6. Run the smoke-test job",
]

_PROSE_LINES = [
    "OCR systems convert images of text into machine-readable",
    "characters. Modern document AI models extend this idea to",
    "preserve layout and structure: tables become markdown tables,",
    "lists keep their ordering, and forms surface as key:value pairs.",
]

_PDF_PAGES = [
    [
        "Research Memo",
        "Author: A. Turing",
        "Subject: On Computable Numbers",
        "Date: November 1936",
        "",
        "Abstract: We sketch a universal computing machine and",
        "show that the Entscheidungsproblem cannot be solved.",
        "Equation: E = m * c^2",
    ],
    [
        "Page 2 of 2",
        "Conclusion: the halting problem is undecidable.",
        "References:",
        "[1] Turing 1936",
        "[2] Church 1936",
    ],
]


EVAL_RECORDS: list[dict[str, Any]] = [
    {
        "inputs": {
            "image_b64": _png(_PROSE_LINES),
            "description": "running prose paragraph",
            "doc_type": "prose",
        },
        "expectations": {
            "expected_response": "\n".join(_PROSE_LINES),
            "numeric_tokens": [],
            "ordered_lines": _PROSE_LINES,
        },
    },
    {
        "inputs": {
            "image_b64": _png(_RECEIPT_LINES),
            "description": "store receipt with currency totals",
            "doc_type": "receipt",
        },
        "expectations": {
            "expected_response": "\n".join(_RECEIPT_LINES),
            "numeric_tokens": [
                "2026-04-15", "4421", "$14.99", "$7.50", "$29.95",
                "$52.44", "$4.33", "$56.77", "8.25%",
            ],
            "ordered_lines": [
                "ACME HARDWARE", "Subtotal", "Tax", "Total",
            ],
        },
    },
    {
        "inputs": {
            "image_b64": _png(_TABLE_LINES),
            "description": "4x4 numeric table",
            "doc_type": "table",
        },
        "expectations": {
            "expected_response": "\n".join(_TABLE_LINES),
            "numeric_tokens": [
                "120.5", "138.2", "151.7", "88.1", "95.4", "102.9",
                "142.0", "150.8", "167.3", "76.6", "81.2", "89.4",
            ],
            "ordered_lines": ["North", "South", "East", "West"],
        },
    },
    {
        "inputs": {
            "image_b64": _png(_FORM_LINES),
            "description": "clinical key:value form",
            "doc_type": "form",
        },
        "expectations": {
            "expected_response": "\n".join(_FORM_LINES),
            "numeric_tokens": ["1984-07-12", "00457821", "10mg"],
            "ordered_lines": [
                "Patient:", "MRN:", "Diagnosis:", "Allergies:", "Rx:", "Provider:",
            ],
        },
    },
    {
        "inputs": {
            "image_b64": _png(_LIST_LINES),
            "description": "ordered numbered list (reading order)",
            "doc_type": "list",
        },
        "expectations": {
            "expected_response": "\n".join(_LIST_LINES),
            "numeric_tokens": ["1.", "2.", "3.", "4.", "5.", "6."],
            "ordered_lines": [
                "1. Run databricks bundle validate",
                "2. Run databricks bundle deploy",
                "3. Run the log-ocr-models job",
                "4. Resolve aliases",
                "5. Apply the SQL UDFs",
                "6. Run the smoke-test",
            ],
        },
    },
    {
        "inputs": {
            "image_b64": _multipage_pdf(_PDF_PAGES),
            "description": "two-page scientific PDF with an equation",
            "doc_type": "pdf",
        },
        "expectations": {
            "expected_response": "\n".join(line for page in _PDF_PAGES for line in page),
            "numeric_tokens": ["1936", "E = m"],
            "ordered_lines": [
                "Research Memo", "Abstract", "Conclusion", "References",
            ],
        },
    },
]

print(f"Built {len(EVAL_RECORDS)} eval records.")
for r in EVAL_RECORDS:
    print(f"  - {r['inputs']['doc_type']:8s}  {r['inputs']['description']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Endpoint client + `predict_fn` factory

# COMMAND ----------


_w = WorkspaceClient()


def _extract_text(predictions: Any) -> str:
    """Endpoints return a list of strings (or list of {"output": str})."""
    if predictions is None:
        return ""
    if isinstance(predictions, list) and predictions:
        first = predictions[0]
        if isinstance(first, dict):
            return str(first.get("output", "") or "")
        return str(first)
    if isinstance(predictions, dict):
        return str(predictions.get("output", "") or "")
    return str(predictions)


def make_predict_fn(endpoint: str):
    """Closure bound to one serving endpoint."""

    @mlflow.trace(name=f"call_{endpoint}")
    def predict_fn(
        image_b64: str,
        description: str | None = None,
        doc_type: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        try:
            resp = _w.serving_endpoints.query(
                name=endpoint,
                dataframe_records=[{"image_b64": image_b64}],
            )
            preds = resp.predictions if hasattr(resp, "predictions") else resp
            text = _extract_text(preds)
            error: str | None = None
        except Exception as exc:  # noqa: BLE001
            text = ""
            error = str(exc)[:600]
        elapsed = time.time() - t0
        return {
            "response": text,
            "latency_s": round(elapsed, 3),
            "error": error,
            "endpoint": endpoint,
            "doc_type": doc_type,
        }

    return predict_fn


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. OCR-focused scorers

# COMMAND ----------


_WS_RE = re.compile(r"\s+")
_TABLE_DELIMS = ("|", "---", "═", "│")


def _normalise(s: str) -> str:
    """Collapse whitespace + lowercase. Used for CER/WER/recall, NOT for the
    raw text similarity scorer (which preserves case)."""
    return _WS_RE.sub(" ", (s or "")).strip().lower()


def _tokens(s: str) -> list[str]:
    return _normalise(s).split()


@scorer
def cer(outputs: dict[str, Any], expectations: dict[str, Any]) -> Feedback:
    """Character Error Rate -- canonical OCR metric. Lower is better."""
    ref = _normalise(expectations.get("expected_response") or "")
    hyp = _normalise(outputs.get("response") or "")
    if not ref:
        return Feedback(value=None, rationale="No expected_response provided.")
    dist = Levenshtein.distance(ref, hyp)
    score = dist / max(len(ref), 1)
    return Feedback(
        value=round(score, 4),
        rationale=f"edit_distance={dist}  ref_len={len(ref)}  hyp_len={len(hyp)}",
    )


@scorer
def char_accuracy(outputs: dict[str, Any], expectations: dict[str, Any]) -> Feedback:
    """`1 - cer`, capped at [0, 1]. Aggregates upwards so charts read intuitively."""
    ref = _normalise(expectations.get("expected_response") or "")
    hyp = _normalise(outputs.get("response") or "")
    if not ref:
        return Feedback(value=None, rationale="No expected_response provided.")
    dist = Levenshtein.distance(ref, hyp)
    val = max(0.0, 1.0 - dist / max(len(ref), 1))
    return Feedback(value=round(val, 4))


@scorer
def wer(outputs: dict[str, Any], expectations: dict[str, Any]) -> Feedback:
    """Word Error Rate over whitespace tokens. Lower is better."""
    ref = _tokens(expectations.get("expected_response") or "")
    hyp = _tokens(outputs.get("response") or "")
    if not ref:
        return Feedback(value=None, rationale="No expected_response provided.")
    # Token-level Levenshtein via rapidfuzz: distance() works on any sequence.
    dist = Levenshtein.distance(ref, hyp)
    score = dist / max(len(ref), 1)
    return Feedback(
        value=round(score, 4),
        rationale=f"word_edit_distance={dist}  ref_words={len(ref)}  hyp_words={len(hyp)}",
    )


@scorer
def text_similarity(outputs: dict[str, Any], expectations: dict[str, Any]) -> Feedback:
    """rapidfuzz token_set_ratio in [0, 1]. Robust to minor reordering and
    page-separator noise that CER over-penalises."""
    ref = expectations.get("expected_response") or ""
    hyp = outputs.get("response") or ""
    if not ref:
        return Feedback(value=None)
    return Feedback(value=round(token_set_ratio(ref, hyp) / 100.0, 4))


@scorer
def numeric_fidelity(outputs: dict[str, Any], expectations: dict[str, Any]) -> Feedback:
    """Fraction of must-keep numeric tokens (currency, dates, IDs, percentages)
    that appear *exactly* (case-insensitive) in the OCR output. The single
    metric SAs care about most for receipts / invoices / clinical forms."""
    tokens: list[str] = list(expectations.get("numeric_tokens") or [])
    if not tokens:
        return Feedback(value=None, rationale="No numeric_tokens for this doc.")
    text = _normalise(outputs.get("response") or "")
    hits = [t for t in tokens if t.lower() in text]
    misses = [t for t in tokens if t.lower() not in text]
    val = len(hits) / len(tokens)
    return Feedback(
        value=round(val, 4),
        rationale=f"{len(hits)}/{len(tokens)} numeric tokens preserved.  missing={misses}",
    )


@scorer
def reading_order(outputs: dict[str, Any], expectations: dict[str, Any]) -> Feedback:
    """Are `ordered_lines` present AND in the same order in the output?"""
    expected: list[str] = list(expectations.get("ordered_lines") or [])
    if not expected:
        return Feedback(value=None)
    text = _normalise(outputs.get("response") or "")
    cursor = 0
    in_order = 0
    found = 0
    for line in expected:
        needle = _normalise(line)
        idx = text.find(needle, cursor)
        if idx != -1:
            found += 1
            in_order += 1
            cursor = idx + len(needle)
        else:
            # Present but out of order? Count as found-but-not-in-order.
            if needle in text:
                found += 1
    val = in_order / len(expected)
    return Feedback(
        value=round(val, 4),
        rationale=(
            f"{in_order}/{len(expected)} lines appeared in order; "
            f"{found}/{len(expected)} appeared anywhere."
        ),
    )


@scorer
def table_structure(outputs: dict[str, Any], inputs: dict[str, Any]) -> Feedback:
    """For the table doc, did the OCR emit a markdown-style table?

    Returns None for non-table rows so the metric only aggregates over the
    relevant subset."""
    if (inputs.get("doc_type") or "") != "table":
        return Feedback(value=None, rationale="Not a table document.")
    text = outputs.get("response") or ""
    has_pipes = sum(1 for ln in text.splitlines() if ln.count("|") >= 2) >= 2
    has_separator = any(d in text for d in _TABLE_DELIMS[1:]) or "---" in text
    val = float(has_pipes and has_separator) if has_pipes else 0.0
    return Feedback(
        value=val,
        rationale=f"pipe_rows>=2={has_pipes}  separator={has_separator}",
    )


@scorer
def latency_s(outputs: dict[str, Any]) -> Feedback:
    val = outputs.get("latency_s")
    return Feedback(value=float(val) if val is not None else None)


@scorer
def non_empty(outputs: dict[str, Any]) -> Feedback:
    err = outputs.get("error")
    text = outputs.get("response") or ""
    if err:
        return Feedback(value=False, rationale=f"Endpoint errored: {err[:200]}")
    return Feedback(value=bool(text.strip()), rationale=f"len_chars={len(text)}")


# OCR-specific LLM judge. Uses a transcription rubric rather than the generic
# `Correctness` judge -- the prompt explicitly asks about character accuracy,
# numeric fidelity, reading order, and structure preservation.
#
# `make_judge` only accepts the top-level template vars {inputs, outputs,
# expectations, trace, conversation}; it does NOT support dotted access
# (e.g. `{{ inputs.doc_type }}` -> error). We pass the whole dicts and let
# the judge introspect them.
ocr_quality = make_judge(
    name="ocr_quality",
    instructions=(
        "You are evaluating an OCR / document-parsing system.\n\n"
        "INPUTS (includes `doc_type` and `description` of the source document):\n"
        "{{ inputs }}\n\n"
        "EXPECTATIONS (the `expected_response` field is the ground-truth text):\n"
        "{{ expectations }}\n\n"
        "OUTPUTS (the `response` field is the system's transcription):\n"
        "{{ outputs }}\n\n"
        "Compare `outputs.response` against `expectations.expected_response`.\n"
        "Rate the transcription on these four dimensions, weighted equally:\n"
        "  1. Character accuracy   (letters, digits, punctuation correct)\n"
        "  2. Numerical fidelity   (currency, dates, IDs preserved exactly)\n"
        "  3. Reading order        (lines/paragraphs in source order)\n"
        "  4. Structure            (tables -> markdown tables, lists -> lists,\n"
        "                           key:value pairs preserved)\n\n"
        "Respond with EXACTLY one of these strings:\n"
        "  'excellent'   (publication-quality transcription)\n"
        "  'good'        (minor errors, fully usable)\n"
        "  'fair'        (noticeable errors, partially usable)\n"
        "  'poor'        (unusable; major content missing or garbled)"
    ),
    model=JUDGE_MODEL,
)


SCORERS = [
    cer,
    char_accuracy,
    wer,
    text_similarity,
    numeric_fidelity,
    reading_order,
    table_structure,
    ocr_quality,
    latency_s,
    non_empty,
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run evaluation: one named run per endpoint

# COMMAND ----------


per_endpoint_metrics: dict[str, dict[str, float]] = {}

for endpoint in ENDPOINTS:
    print(f"\n=== Evaluating {endpoint} ===")
    with mlflow.start_run(run_name=endpoint):
        mlflow.set_tag("endpoint", endpoint)
        mlflow.set_tag("dataset_size", len(EVAL_RECORDS))
        mlflow.set_tag("eval_kind", "ocr-document-parsing")
        try:
            results = mlflow.genai.evaluate(
                data=EVAL_RECORDS,
                predict_fn=make_predict_fn(endpoint),
                scorers=SCORERS,
            )
            metrics = dict(results.metrics or {})
            per_endpoint_metrics[endpoint] = metrics
            print(f"run_id={results.run_id}")
            for k, v in sorted(metrics.items()):
                print(f"  {k}: {v}")
        except Exception as exc:
            print(f"!! evaluate() raised: {exc!r}")
            per_endpoint_metrics[endpoint] = {"error": str(exc)[:500]}
            mlflow.set_tag("eval_error", str(exc)[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Side-by-side OCR scoreboard

# COMMAND ----------


_PRIMARY = [
    "char_accuracy/mean",
    "cer/mean",
    "wer/mean",
    "text_similarity/mean",
    "numeric_fidelity/mean",
    "reading_order/mean",
    "table_structure/mean",
    "latency_s/mean",
    "non_empty/mean",
]

rows: list[dict[str, Any]] = []
for endpoint, metrics in per_endpoint_metrics.items():
    row: dict[str, Any] = {"endpoint": endpoint}
    for key in _PRIMARY:
        v = metrics.get(key)
        # Cast numpy scalars (float64 etc.) to native python so spark schema
        # inference doesn't choke; keep None as None.
        row[key] = float(v) if isinstance(v, (int, float)) else v
    rows.append(row)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if v is None:
        return "-"
    return str(v)


print("\nOCR scoreboard (means across the eval set; lower is better for cer/wer/latency_s):")
header = ["endpoint", *_PRIMARY]
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")
for r in rows:
    cells = [_fmt(r.get(h)) for h in header]
    print("| " + " | ".join(cells) + " |")

# Spark + display() are only available inside a Databricks notebook context;
# guard so the same file also runs as a plain Python script.
try:
    import pandas as pd

    pdf = pd.DataFrame(rows)
    try:
        display(spark.createDataFrame(pdf))  # type: ignore[name-defined]  # noqa: F821
    except (NameError, Exception):  # noqa: BLE001 - fall back to pandas display
        try:
            display(pdf)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            print(pdf.to_string(index=False))
except Exception as exc:  # noqa: BLE001
    print(f"(scoreboard render failed: {exc!r}; metrics still logged to MLflow)")

# Surface a failure if any endpoint returned only empty / errored outputs --
# makes the eval job a useful CI signal, not just a notebook.
broken = [
    ep
    for ep, m in per_endpoint_metrics.items()
    if (m.get("non_empty/mean") or 0) == 0 or "error" in m
]
if broken:
    raise AssertionError(f"Endpoints with no successful outputs: {broken}")

print("\nAll endpoints produced output. See the MLflow experiment for traces.")
