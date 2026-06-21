# Reproduce the CBS engine from scratch (HF/vLLM enrichment)

End-to-end steps to rebuild everything on a fresh machine (e.g. neumann). The
StatLine table **enrichment runs from a HuggingFace model via vLLM** (GPU),
instead of local Ollama. Interactive pieces (chat agent, publication LLM
extraction) still talk to an Ollama endpoint — see notes at the end.

Much of the processed data is already committed, so after `git pull` you can
**skip the slow steps** and jump to "Build the index". The commands below are the
full from-scratch path.

## 0. Clone + environment
```bash
git clone -b cbs-intelligence-engine https://github.com/avishekanand/open-govt-data.git
cd open-govt-data
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # CPU deps (pandas, pyarrow, streamlit, matplotlib...)
pip install -r requirements-gpu.txt        # GPU: vllm + huggingface_hub  (GPU node only)
```

## 1. StatLine catalogue + metadata (CPU, no model)
```bash
python -m cbs.catalog                                   # 4,857-table catalogue
python -m cbs.batch_ingest_statline --limit 1277 --sample-data 5   # metadata for active tables
```
Outputs → `data/processed/statline_catalog.parquet`, `data/processed/catalog_meta/*.parquet`.
(Already in git — skip if you pulled.)

## 2. Enrichment from HuggingFace via vLLM (GPU)
Cache the model on shared storage so it downloads once:
```bash
export HF_HOME=/path/to/shared/hf            # e.g. an umbrella/scratch share
# export HF_TOKEN=hf_xxx                      # only for gated models (e.g. gemma)

python -m cbs.enrich_cbs_vllm \
  --model Qwen/Qwen2.5-7B-Instruct \          # ungated, strong JSON. or google/gemma-2-9b-it (gated)
  --output data/processed/cbs_enriched_vllm.jsonl \
  --resume --tensor-parallel-size 1          # raise TP for multi-GPU
```
On SLURM (DAIC-style), the whole pipeline is wrapped:
```bash
sbatch scripts/enrich_daic.slurm             # edit --partition + HF_HOME first
```

Each model/machine should write its **own** file (so runs don't clobber each other),
e.g. `--output data/processed/cbs_enriched_<model>_<host>.jsonl`. The index merges
them all automatically (next step) — no symlink needed.

## 3. Build the search index (CPU)
```bash
python -m cbs.build_search_index --selftest          # -> data/processed/cbs_search.db
```
By default it **merges every `data/processed/cbs_enriched_*.jsonl`** (highest
self-reported confidence wins on duplicate tables), so a gemma4 file and a qwen
file are combined. To index a specific subset instead:
```bash
python -m cbs.build_search_index --enriched data/processed/cbs_enriched_qwen_neumann.jsonl
```

## 4. Apps
```bash
# Search — no model needed:
streamlit run cbs/search_app.py --server.port 8501
# Chat agent — needs an Ollama endpoint (see notes):
OLLAMA_HOST=http://<host>:11434 streamlit run cbs/chat_app.py --server.port 8502
```
View remotely:  `ssh -L 8501:localhost:8501 neumann`  then open `localhost:8501`.

## 5. Publication-evidence layer
Copy the workbook (not in git) to `data/raw/Publications_overview_internet_May_26.xlsx`, then:
```bash
python -m cbs.pub_ingest                              # workbook -> records + URLs  (CPU)
python -m cbs.pub_download                            # crawl + extract text        (network, resumable)
OLLAMA_HOST=http://<host>:11434 python -m cbs.pub_extract --model qwen2.5:7b   # which CBS datasets used
python -m cbs.pub_report                              # -> data/processed/pub/publication_findings.md
```
Records / URLs / extractions / findings MD are already committed; raw docs + scraped
text + `documents.parquet` are git-ignored (re-crawl with `pub_download`).

## 6. (optional) Agent correctness eval
```bash
OLLAMA_HOST=http://<host>:11434 python -m cbs.eval_agent     # 10 labelled cases, PASS/FAIL + accuracy
```

## Notes — which steps use which model
| Step | Backend | Notes |
|---|---|---|
| Table enrichment (`enrich_cbs_vllm`) | **HF + vLLM** (GPU) | this runbook's path; cached via `HF_HOME` |
| Chat agent / eval (`agent`, `eval_agent`) | **Ollama** | uses `$OLLAMA_HOST`, `MODEL`, `VERIFY_MODEL` (gemma4) |
| Publication extraction (`pub_extract`) | **Ollama** | uses `$OLLAMA_HOST`, `$PUB_MODEL` (qwen2.5) |
| Catalogue / metadata / index / search | none | pure CPU |

The interactive/agent LLM calls go through Ollama's `/api/chat`. On a GPU box you can
either run Ollama there, or point `OLLAMA_HOST` at any machine that has it. (A native
vLLM/HF path for the agent + pub_extract is a possible future addition.)
