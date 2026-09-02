# OGD — Open Government Data metadata engine 🏛️📊

A searchable, LLM-enriched **metadata catalogue** over official statistics from
**Eurostat** and **CBS Netherlands (StatLine)**.

The design principle: index the *metadata*, fetch the *observations* on demand.
A dataset's full structure — every dimension and every category code→label — is
available from a tiny API call, so 12,308 datasets are described in ~88 MB of
metadata instead of the hundreds of GB their observations occupy (Eurostat's
6.2 B values alone are ~207 GB; CBS adds another 13.4 B).

## 📦 What's in the box

| | datasets | category labels |
|---|---|---|
| 🇪🇺 Eurostat | 7,438 (of 7,572 live) | 1,026,725 |
| 🇳🇱 CBS StatLine | 4,870 (all statuses) | 3,148,436 |
| **enriched records** | **12,308** | **4,175,161** |

Enrichment quality (Qwen3-32B via vLLM, one A100-80GB, 3h29 total):

- **100%** of records grounded in real category values
- **98.6%** unique `example_queries` (53,827 / 54,580)
- 0 duplicate array entries · 0.09% invalid join ids · 0 generation failures

## 🚀 Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU
pip install -r requirements-gpu.txt      # GPU node only (vllm)
```

Everything below is idempotent and resumable — re-running skips completed work.

### 1. Harvest metadata (CPU, no model, ~15 min)

```bash
# Eurostat: live TOC + per-dataset SDMX structure (dimension names + codelists)
python -m enrich.ingest_eurostat_meta --workers 8

# CBS: catalogue + semantic layer + dimension code lists
python -m cbs.catalog
python -m cbs.batch_ingest_statline --status Regulier         --limit 4000
python -m cbs.batch_ingest_statline --status Gediscontinueerd --limit 4000
python -m enrich.ingest_cbs_codes --workers 8
```

### 2. Enrich (GPU)

```bash
# Always dry-run first: reports grounding coverage and validates the JSON schema
# against BOTH structured-output backends (see "Gotchas").
python -m enrich.run_vllm --source both --dry-run

# Real run — on SLURM:
sbatch scripts/enrich_unified.slurm
# or directly:
python -m enrich.run_vllm --source both --model Qwen/Qwen3-32B --resume
```

Useful flags: `--limit N` (per source, for smoke tests), `--source cbs|eurostat|both`,
`--chunk-size` (flush interval), `--include-ungrounded`.

### 3. Fetch observations — on demand only

```bash
python -m cbs.fetch_table_data --table 83765NED --max-obs 5000
python eurostat_fetch_one.py --code TPS00001 --out population.csv
```

## 🗂️ The enriched record

One JSON object per line in `data/processed/enriched_unified_qwen3-32b.jsonl`,
with `publisher` (`ESTAT` / `CBS`) discriminating the source. The schema splits
in two halves on purpose:

**Deterministic** — computed from the data, never asked of the model:

```json
"code": "nama_10r_3gdp", "publisher": "ESTAT",
"title_native": "Gross domestic product (GDP) at current market prices by NUTS 3 region",
"coverage": {"start": 2000, "end": 2024, "n_periods": 25},
"dimensions": [{"id": "geo", "name": "Geopolitical entity (reporting)",
                "n_categories": 1814, "sample": ["EU27_2020 (European Union - 27)", "..."]}],
"n_observations": 296322, "grounded": true
```

**Generative** — the doc2query fields, JSON-schema-constrained:

```json
"title_en", "enriched_description", "example_queries", "potential_applications",
"key_dimensions", "topics", "join_keys", "confidence"
```

Earlier pipelines let the model restate coverage and dimensions, and it got them
wrong — echoing a stale catalogue `dataend` when the data ran two years further.
Now it can't: those fields never reach the model's output.

## 📁 Where the data is

```
data/processed/
├── enriched_unified_qwen3-32b.jsonl   ← the artifact: 12,308 enriched records
├── eurostat_metadata.jsonl            7,438 datasets: dims + codelists
├── eurostat_catalog.parquet           7,572 live TOC rows
├── eurostat_metadata.failures.json    134 unreachable (132× HTTP 413, 2× 401)
├── cbs_codelists.jsonl                4,870 tables: dimension code lists
├── statline_catalog.parquet           4,868 CBS tables, all statuses
└── catalog_meta/                      CBS semantic layer
    ├── statline_datasets.parquet      4,870 tables
    ├── statline_dimensions.parquet    13,507 dimensions
    └── statline_measures.parquet      157,954 measures
```

Observation data (`downloads/`, `data/processed/tables/`) is **git-ignored** —
it is not needed to build or search the index.

## 🇳🇱 CBS apps — search, chat, publication evidence

```bash
python -m cbs.build_search_index --selftest        # SQLite FTS5 index
streamlit run cbs/search_app.py --server.port 8501 # search UI
OLLAMA_HOST=http://<host>:11434 streamlit run cbs/chat_app.py --server.port 8502
```

> ⚠️ **Not yet wired to the unified corpus.** `build_search_index` still globs
> `data/processed/cbs_enriched_*.jsonl` and expects the older CBS-only field
> names, so it indexes the superseded gemma4 / qwen2.5 runs and ignores
> `enriched_unified_*.jsonl` — and Eurostat entirely. Porting it to the unified
> schema is the next task.

**Publication evidence layer** — indexes public evidence of how CBS microdata has
been used (never the confidential microdata itself): 3,106 records → 2,217 URLs →
1,487 downloaded → 573 LLM extractions.

```bash
python -m cbs.pub_ingest && python -m cbs.pub_download
OLLAMA_HOST=http://<host>:11434 python -m cbs.pub_extract --model qwen2.5:7b
python -m cbs.pub_report        # -> data/processed/pub/publication_findings.md
```

## ⚠️ Gotchas worth knowing

- **Structured decoding across vLLM versions.** The API was renamed twice
  (`structured_outputs=` ≥0.11, `guided_decoding=` 0.6–0.10, `guided_json=` <0.6).
  `run_vllm` tries newest first and **aborts** rather than silently generating
  unconstrained text (the old code fell through to a regex fallback and dropped
  tables without recording which).
- **Two backends, two schema subsets.** vLLM picks between xgrammar and
  llguidance via `backend='auto'`. `uniqueItems` compiles under xgrammar and
  hard-fails the whole job under llguidance. `--dry-run` validates against both;
  uniqueness is enforced in `finalize_record()` instead.
- **Category sampling is spread, not truncated.** Showing the first 12 of 1,814
  alphabetical NUTS regions produced four example queries about Albanian
  districts. `enrich.sampling.spread_sample` samples across the codelist.
- **Eurostat catalogue titles lost every space** upstream
  (`Long-termresidentsbycitizenshipon31December`). The SDMX `dataset_label` is
  clean and is preferred; catalogue coverage is likewise stale (a dataset
  declaring `dataend: 2023` had data through 2025).
- **Qwen3 is a hybrid reasoning model** — enrichment sets
  `enable_thinking=False`, or it burns the token budget on a `<think>` block.

## 🔎 Coverage gaps (known, not silent)

| gap | count | fix |
|---|---|---|
| Eurostat datasets too large for a 1-period slice (HTTP 413) | 132 | re-fetch sliced by `geo` |
| in the SDMX dataflow registry but absent from the TOC | 32 | ingest directly |
| unauthorized (401) | 2 | likely genuinely restricted |
| invalid join ids (model paraphrased a dimension name) | 65 | fuzzy-match post-pass |

**Checking for completeness:** cross-reference the two independent Eurostat
catalogues. Every servable dataset has a dataflow, so
`sdmx/2.1/dataflow/ESTAT/all/latest` (8,152 entries) is authoritative; the TOC
(7,572) is a strict subset, and 548 of the difference are `$DV_` derived views
of datasets already listed.

## 📚 Legacy Eurostat pipeline (superseded)

`csv_to_ollama_jsonl_complete_only.py` + `data/eurostat_gemma3*.jsonl` predate
this work and are kept for provenance only. They covered 698 of ~7,600 datasets,
and the `_gpt5` variant is a template rather than a refinement — 96 unique
queries across 3,540 (2.7%), the same string repeated 590 times. Use
`enrich.run_vllm` instead.

`batch_fetch_eurostat.py` remains useful for bulk observation downloads, but it
is no longer part of the index pipeline.

## 📄 License

MIT — see [LICENSE](LICENSE). Reproduction steps: [RUNBOOK.md](RUNBOOK.md).

## 🙏 Acknowledgments

**Eurostat** and **CBS Netherlands** for open data APIs · **vLLM** and **Qwen**
for local batched inference · **Ollama** for the interactive agent path.
