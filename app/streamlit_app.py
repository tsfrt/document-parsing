"""Streamlit playground for the OCR endpoints.

Lets you upload a PDF / PNG / JPEG, pick one or more OCR endpoints, and
compare their Markdown output side-by-side with per-model timing. Uses the
service principal credentials injected by Databricks Apps (no secrets to
configure).
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass

import streamlit as st
from databricks.sdk import WorkspaceClient

st.set_page_config(page_title="OCR Document Parsing Playground", layout="wide")


@dataclass(frozen=True)
class Endpoint:
    key: str
    label: str
    name: str


def load_endpoints() -> list[Endpoint]:
    raw = os.environ.get("MODEL_ENDPOINTS")
    if not raw:
        return [
            Endpoint("florence", "Florence-2 large-ft (Microsoft, 770M)", "doc-parser-florence"),
            Endpoint("phi3",     "Phi-3.5-vision (Microsoft, 4.2B)",      "doc-parser-phi3-vision"),
            Endpoint("granite",  "Granite-Vision-3.2 (IBM, 2.5B)",        "doc-parser-granite-vision"),
        ]
    return [Endpoint(**e) for e in json.loads(raw)]


@st.cache_resource
def workspace_client() -> WorkspaceClient:
    return WorkspaceClient()


def query_endpoint(client: WorkspaceClient, endpoint: Endpoint, image_b64: str,
                   prompt: str | None, output_format: str) -> tuple[str, float, str | None]:
    record: dict[str, object] = {"image_b64": image_b64, "output_format": output_format}
    if prompt:
        record["prompt"] = prompt
    t0 = time.perf_counter()
    try:
        resp = client.serving_endpoints.query(
            name=endpoint.name,
            dataframe_records=[record],
        )
    except Exception as exc:
        return "", time.perf_counter() - t0, str(exc)
    elapsed = time.perf_counter() - t0
    preds = getattr(resp, "predictions", None) or resp
    if isinstance(preds, list) and preds:
        return str(preds[0]), elapsed, None
    return "", elapsed, "No predictions returned"


# ---------------------------------------------------------------------- UI ----

ENDPOINTS = load_endpoints()
client = workspace_client()

st.title("OCR Document Parsing Playground")
st.caption(
    "Upload a PDF or image, choose one or more OCR endpoints, and compare results. "
    "Cold-start may take 1-3 minutes per endpoint when scaled to zero."
)

with st.sidebar:
    st.header("Settings")
    chosen = st.multiselect(
        "Models",
        options=[ep.key for ep in ENDPOINTS],
        default=[ENDPOINTS[0].key],
        format_func=lambda k: next(e.label for e in ENDPOINTS if e.key == k),
    )
    output_format = st.radio("Output format", ["markdown", "json"], horizontal=True)
    prompt = st.text_area(
        "Custom prompt (optional)",
        placeholder="Leave blank to use each model's default prompt.",
        height=100,
    )
    st.markdown("---")
    st.caption("Endpoints are picked up from the `MODEL_ENDPOINTS` env var.")


uploaded = st.file_uploader(
    "Upload a document",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

raw = uploaded.read()
image_b64 = base64.b64encode(raw).decode()
st.write(f"**File:** `{uploaded.name}` &nbsp; **Size:** {len(raw):,} bytes")

if not chosen:
    st.warning("Pick at least one model in the sidebar.")
    st.stop()

if not st.button("Parse", type="primary"):
    st.stop()

selected = [ep for ep in ENDPOINTS if ep.key in chosen]
results: dict[str, tuple[str, float, str | None]] = {}

with st.spinner(f"Parsing with {len(selected)} model(s)..."):
    for ep in selected:
        results[ep.key] = query_endpoint(
            client, ep, image_b64, prompt or None, output_format
        )

cols = st.columns(len(selected))
for col, ep in zip(cols, selected):
    text, elapsed, err = results[ep.key]
    with col:
        st.subheader(ep.label)
        st.caption(f"`{ep.name}` &middot; {elapsed:0.1f}s")
        if err:
            st.error(f"Endpoint error: {err}")
            continue
        if not text:
            st.warning("Empty response.")
            continue
        if output_format == "markdown":
            st.markdown(text)
        else:
            try:
                st.json(json.loads(text))
            except json.JSONDecodeError:
                st.code(text)
        st.download_button(
            "Download",
            data=text,
            file_name=f"{uploaded.name}.{ep.key}.{output_format}.txt",
            mime="text/plain",
            key=f"dl-{ep.key}",
        )
