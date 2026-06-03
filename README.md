# document-parsing

A Databricks Asset Bundle that deploys three GPU-backed OCR Model Serving endpoints — **Florence-2 large-ft**, **Phi-3.5-vision**, and **Granite-Vision-3.2** — and exposes them through SQL UDFs callable via `ai_query` for natural use in Databricks SQL pipelines. All three models are USA-derived and sized to fit on a single 16 GB GPU (`GPU_SMALL` / T4-class). A minimal Streamlit playground app lets you upload a PDF/image and compare output across models.

## What you get

| Resource | What it is |
|---|---|
| `doc-parser-florence` | Custom pyfunc serving endpoint, `GPU_SMALL`, scale-to-zero, wraps [`microsoft/Florence-2-large-ft`](https://huggingface.co/microsoft/Florence-2-large-ft) (770M). Microsoft's compact task-prompted VLM with a dedicated `<OCR>` task head; the smallest, fastest-to-boot model in the lineup. Loaded in fp16 (~3 GB). MIT-licensed, non-gated. |
| `doc-parser-phi3-vision` | Wraps [`microsoft/Phi-3.5-vision-instruct`](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) (4.2B). Strong general multimodal OCR + reasoning. Loaded in bf16 (~9 GB). |
| `doc-parser-granite-vision` | Wraps [`ibm-granite/granite-vision-3.2-2b`](https://huggingface.co/ibm-granite/granite-vision-3.2-2b) (2.5B). IBM's compact document-understanding VLM (tables, charts, infographics). Loaded in bf16 (~5 GB). |
| SQL UDFs | `parse_doc_florence`, `parse_doc_phi3`, `parse_doc_granite`, plus a router `parse_doc(content, model)` |
| Streamlit app | `doc-parser-playground` Databricks App for interactive comparison |

### Picking a model

| Use case | Recommended model |
|---|---|
| Mixed PDFs + photos, want a fast default | **Florence-2 large-ft** (smallest, sub-second inference; default prompt is `<OCR>`) |
| General-purpose OCR with strong reasoning over content (Q&A on parsed output) | **Phi-3.5-vision** |
| Tables, charts, forms, infographics | **Granite-Vision-3.2** |

## Quick SQL example

```sql
-- Parse all PDFs/images in a Volume with one model
SELECT path, main.doc_parsing.parse_doc_florence(content) AS markdown
FROM read_files(
  '/Volumes/main/doc_parsing/inbox/',
  format => 'binaryFile',
  fileNamePattern => '*.{pdf,png,jpg,jpeg,PDF,PNG,JPG,JPEG}'
);

-- Compare two models side-by-side
SELECT path,
       main.doc_parsing.parse_doc(content, 'florence') AS via_florence,
       main.doc_parsing.parse_doc(content, 'granite')  AS via_granite
FROM read_files('/Volumes/main/doc_parsing/inbox/', format => 'binaryFile');
```

See [`sql/examples.sql`](sql/examples.sql) for more patterns including direct `ai_query` calls with custom prompts and JSON output.

## Why a STRUCT input (not `files =>`)

Databricks `ai_query` supports a `files => content` parameter for binary image input, but that path is documented for the *Foundation Model APIs* (multimodal chat models that follow the OpenAI `image_url` schema, JPEG/PNG only). For custom serving endpoints, the canonical input is a `STRUCT` (passed as a Pandas DataFrame to the pyfunc). We use that:

```sql
ai_query('doc-parser-florence',
         named_struct('image_b64', base64(content)),
         failOnError => false)
```

This lets us:
- Accept PDFs as well as PNG/JPEG (PDFs are rasterized inside the pyfunc with `pypdfium2`).
- Pass optional `prompt`, `output_format` (`markdown` | `json`), and `max_pages` fields.
- Return per-row error envelopes instead of failing the whole batch.

## Architecture

```
┌────────────────────┐    SQL UDF      ┌────────────────────────────┐
│ ai_query / SQL UDF │ ──────────────► │ Model Serving (GPU_SMALL)  │
└────────────────────┘                 │  doc-parser-florence       │
        ▲                              │  doc-parser-phi3-vision    │
        │                              │  doc-parser-granite-vision │
┌────────────────────┐  REST            │  ▲                         │
│ Streamlit playground│ ────────────────┘  │                         │
└────────────────────┘                  loads weights from           │
                                        Unity Catalog registered     │
                                        models                       │
                                        (main.doc_parsing.*)         │
                                                                     │
        Hugging Face ─── snapshot_download ── log_models.py ─────────┘
```

## Repository layout

```
ai_parse_doc/
├── databricks.yml                # bundle root
├── pyproject.toml                # ruff + pytest config, dependency groups
├── resources/
│   ├── endpoints.yml             # 3 serving endpoints + 3 registered models
│   ├── jobs.yml                  # log, deploy-udfs, smoke-test jobs
│   ├── volumes.yml               # schema + volumes (inbox, sample_docs)
│   └── app.yml                   # Streamlit Databricks App resource
├── sql/
│   ├── udfs.sql                  # parse_doc_* UDFs (deployed by job)
│   └── examples.sql
├── src/doc_parser/
│   ├── base.py                   # OcrPyfunc base
│   ├── pdf_utils.py              # PDF/image decoding helpers
│   ├── log_models.py             # log all 3 models to UC + set Production alias
│   └── models/
│       ├── florence.py
│       ├── phi3_vision.py
│       └── granite_vision.py
├── src/tests/
│   ├── test_pdf_utils.py
│   └── test_pyfunc_contracts.py
├── notebooks/
│   ├── 01_log_models.py
│   ├── 02_smoke_test.py
│   └── 04_eval.py                # mlflow.genai.evaluate() across all 3 endpoints
├── app/
│   ├── app.yaml
│   ├── streamlit_app.py
│   └── requirements.txt
└── .github/workflows/
    ├── pr-checks.yml             # ruff + pytest + bundle validate
    ├── pr-preview.yml            # deploy preview bundle on PR
    └── release.yml               # tag v* on main → prod deploy
```

## Deploying

### Prerequisites
- A Unity Catalog–enabled workspace with Model Serving and AI Functions enabled.
- A Pro or Serverless SQL warehouse (for `ai_query` and the UDFs).
- `databricks` CLI (>= 0.235), `jq`, Python 3.11+.
- `DATABRICKS_HOST` + `DATABRICKS_TOKEN` configured locally (or via `databricks auth login`).

### First-time deploy (manual)

```bash
# 1. One-shot deploy — creates schema, volumes, registered_models, jobs, app.
#    Endpoints will be created with version "1" placeholders that don't yet exist.
databricks bundle deploy --target dev

# 2. Run the log job (snapshots weights, logs all 4 models, sets Production alias).
#    This is a long-running task on a GPU cluster; expect ~30-60 min.
databricks bundle run log-ocr-models --target dev

# 3. Resolve aliases and re-deploy so endpoints pick up the actual versions.
for model in florence phi3_vision granite_vision; do
  ver=$(databricks model-versions get-by-alias "main.doc_parsing.${model}" Production --output json | jq -r .version)
  export "BUNDLE_VAR_${model}_version=${ver}"
done
databricks bundle deploy --target dev

# 4. Apply the SQL UDFs.
databricks bundle run deploy-ocr-udfs --target dev

# 5. Smoke-test all three endpoints.
databricks bundle run smoke-test-ocr-endpoints --target dev
```

### Promoting a new model version

1. Re-run `log-ocr-models` (or just one model with `--params models=florence`). The job sets the `Production` alias on the new version.
2. Re-run the `release.yml` workflow (or repeat steps 3–5 above) to roll the endpoints forward.

## CI/CD

Three workflows in `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `pr-checks.yml` | PR / push to main | `ruff check`, `pytest`, `databricks bundle validate --target preview` |
| `pr-preview.yml` | PR opened/updated against main | Deploys a per-branch preview bundle (`preview_<branch>_*`) and posts a PR comment |
| `release.yml` | tag `v*` on main | Two-pass deploy to prod, runs log job, applies UDFs, runs smoke test |

### Required GitHub secrets / variables

| Name | Type | Used by |
|---|---|---|
| `DATABRICKS_HOST` | secret | all three workflows |
| `DATABRICKS_TOKEN` | secret | all three workflows |
| `PROD_CATALOG` | env var (production environment) | `release.yml` |
| `PROD_SCHEMA` | env var (production environment) | `release.yml` |
| `PROD_WAREHOUSE_ID` | env var (production environment) | `release.yml` |

### Branch protection

Configure on the GitHub repo:
- `main` is protected.
- Pull requests require **at least one approval** before merge.
- Required status checks: `lint-and-test`, `bundle-validate`.
- Direct pushes to `main` are disabled — releases happen only via merged PRs and tags.

## Cold start, cost, and capacity caveats

- **Cold start ~60-180s** when the endpoint scales from zero (GPU container build + weight load). Baked-in artifacts (no HF download at startup) keep this predictable. To eliminate cold start, set `scale_to_zero_enabled=false` (one-line bundle var override).
- **VRAM headroom on GPU_SMALL (16 GB T4-class).** Per-model resident weights: Florence-2 (fp16) ~3 GB, Phi-3.5-vision (bf16) ~9 GB, Granite-Vision-3.2 (bf16) ~5 GB. With image-encoder activations and KV cache for 2-4 K-token outputs, every model has comfortable headroom. If you swap in a larger model, move to `GPU_MEDIUM` (24 GB) or NF4-quantize the weights at log time.
- **300 s model-serving cold-load budget.** Larger checkpoints (e.g. 7 B+ models that ship 14 GB of safetensors) reproducibly time out streaming weights from the artifact store on first boot. The lineup above is sized to fit comfortably under this limit. If you experiment with bigger models, plan to bump to `GPU_MEDIUM`/`GPU_LARGE` *and* pre-quantize weights, or pin `scale_to_zero_enabled=false` so warm replicas absorb subsequent deploys.
- **GPU pricing** applies even when `scale_to_zero_enabled=true`, while warm. Tag your endpoints (already done) and review usage via `system.serving.endpoint_usage`.

## Model licenses

Each Hugging Face model is shipped under its own license. Re-host responsibly.

| Model | License | Origin |
|---|---|---|
| Florence-2 large-ft (770M) | MIT | [Microsoft Research](https://www.microsoft.com/research) (Redmond, USA) |
| Phi-3.5-vision-instruct | MIT | [Microsoft Research](https://www.microsoft.com/research) (Redmond, USA) |
| Granite-Vision-3.2-2B | Apache 2.0 | [IBM Research](https://research.ibm.com) (Yorktown Heights, USA) |

## Project rules and intentional deviation

This project follows the team rules around Databricks Asset Bundles + GitHub Actions deployment. **It intentionally skips the Lakebase + Prisma database-branching parts of the rules** because the playground app is stateless (no DB). If a database is added later — for example to store parse history or user feedback — the right follow-up is:

- Add `resources/lakebase.yml` for a Lakebase Postgres instance with a synced production branch.
- Add a `prisma/` directory with a `schema.prisma` and migrations.
- Have `pr-preview.yml` create a feature-branch DB clone (Lakebase branch) before `bundle deploy`.
- Have `release.yml` run `prisma migrate deploy` against the production branch before swapping endpoint versions.

## Out of scope (deliberate)

- Bounding-box / layout JSON output. Easy follow-up: extend `output_format` per model. Granite-Vision-3.2 has a `<doctags>` mode that emits structured layout, and Phi-3.5-vision can be prompted to return JSON.
- Full Lakeflow Declarative Pipeline for batch document ingestion. The provided SQL examples handle ad-hoc batch usage; a streaming pipeline would be a 1-day extension.
- A full OmniDocBench-style evaluation harness. The repo *does* ship a small OCR-focused eval at [`notebooks/04_eval.py`](notebooks/04_eval.py) (run via the `eval-ocr-endpoints` bundle job). It sends 6 curated synthetic documents — prose paragraph, numeric receipt, markdown table, key:value form, ordered list, and a 2-page scientific PDF with an equation — to each endpoint and runs `mlflow.genai.evaluate()` with OCR-specific scorers: `cer` and `char_accuracy` (Levenshtein-based Character Error Rate), `wer` (Word Error Rate), `text_similarity` (rapidfuzz token-set ratio), `numeric_fidelity` (currency/date/ID survival), `reading_order` (lines appear in source order), `table_structure` (markdown table emitted for the table doc), `ocr_quality` (custom `make_judge` with an OCR rubric — character accuracy, numeric fidelity, reading order, structure), plus `latency_s` and `non_empty`. One named MLflow run per endpoint, all under `/Shared/ai-parse-doc/eval` (override via `--var eval_experiment_path=...`). Scaling this to OmniDocBench is a follow-up.
- Specialised LaTeX-aware OCR for academic papers. An earlier iteration of this bundle included `doc-parser-nougat` (`facebook/nougat-base`, 350M), but Nougat is hard-tuned for arXiv-shaped scientific PDFs and falls into a `## References` hallucination mode on anything else. It was dropped to keep the lineup focused on general-purpose document parsing. If you need it back, restore the entry in `REGISTRY` (`src/doc_parser/log_models.py`) and the `doc-parser-nougat` block in `resources/endpoints.yml`.
