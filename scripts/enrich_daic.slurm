#!/usr/bin/env bash
#SBATCH --job-name=cbs-enrich
#SBATCH --partition=gpu               # adjust to your DAIC GPU partition (check `sinfo`)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# CBS doc2query enrichment on TU Delft DAIC — SERVER-FREE batched inference.
#
# No Ollama daemon. vLLM loads a HuggingFace model straight onto the GPU and
# runs ALL table prompts as one batched job. The model is downloaded from HF
# once and cached on the umbrella share (HF_HOME) for subsequent runs.
#
# Submit:   sbatch scripts/enrich_daic.slurm
# Tail:     tail -f slurm-cbs-enrich-<jobid>.out
#
set -euo pipefail

# --- config ------------------------------------------------------------------
REPO="${REPO:-$HOME/projects/open-govt-data}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"   # ungated, strong JSON. gemma needs HF_TOKEN.
LIMIT="${LIMIT:-}"                             # empty = all ingested tables
# Cache HF model weights on the SIR umbrella share so re-runs skip the download.
export HF_HOME="${HF_HOME:-/tudelft.net/staff-umbrella/sir/hf}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
# For gated models (e.g. google/gemma-2-9b-it) export HF_TOKEN before sbatch.
# export HF_TOKEN=hf_xxx

cd "$REPO"
mkdir -p "$HF_HOME"

# --- environment -------------------------------------------------------------
module load miniconda3 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt          # pandas/pyarrow/etc (CPU side)
pip install -q -r requirements-gpu.txt       # vllm + huggingface_hub

# --- 1. ensure metadata exists (catalogue + per-table semantic layer) --------
# CPU-only; skips work already on disk. Set LIMIT to bound the table count.
python -m cbs.catalog
python -m cbs.batch_ingest_statline ${LIMIT:+--limit "$LIMIT"} --sample-data 5

# --- 2. batched enrichment on the GPU (no server) ----------------------------
echo "Enriching with $MODEL on GPU ${CUDA_VISIBLE_DEVICES:-?} ..."
python -m cbs.enrich_cbs_vllm \
  --model "$MODEL" \
  --output data/processed/cbs_enriched_vllm.jsonl \
  ${LIMIT:+--limit "$LIMIT"} \
  --resume \
  --tensor-parallel-size "${TP_SIZE:-1}"

# --- 3. build the search index (point it at the vLLM output) -----------------
# build_search_index reads cbs_enriched_gemma4.jsonl by default; symlink so the
# index picks up the vLLM enrichment without code changes.
ln -sf cbs_enriched_vllm.jsonl data/processed/cbs_enriched_gemma4.jsonl
python -m cbs.build_search_index --selftest

echo "Done. data/processed/cbs_enriched_vllm.jsonl + cbs_search.db are ready."
echo "Serve the UI with:  PORT=8080 ./scripts/serve_app.sh"
