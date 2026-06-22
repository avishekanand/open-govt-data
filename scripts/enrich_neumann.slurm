#!/usr/bin/env bash
#SBATCH --job-name=cbs-enrich
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# Neumann variant of scripts/enrich_daic.slurm.
# Writes enrichment to a model-specific file and does NOT overwrite the
# git-tracked data/processed/cbs_enriched_gemma4.jsonl. No index rebuild.

set -euo pipefail

REPO="${REPO:-$HOME/projects/open-govt-data}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT:-data/processed/cbs_enriched_qwen25_7b_vllm.jsonl}"
LIMIT="${LIMIT:-}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

# This cluster has no CUDA toolkit (no nvcc / no /usr/local/cuda), so FlashInfer
# cannot JIT-compile its sampling kernel at startup. Force vLLM onto its native
# PyTorch top-k/top-p sampler (prebuilt FLASH_ATTN attention is unaffected).
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$REPO"
mkdir -p "$HF_HOME"

module load miniconda3 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r requirements-gpu.txt

python -m cbs.catalog
python -m cbs.batch_ingest_statline ${LIMIT:+--limit "$LIMIT"} --sample-data 5

echo "Enriching with $MODEL on GPU ${CUDA_VISIBLE_DEVICES:-?} -> $OUT"
python -m cbs.enrich_cbs_vllm \
  --model "$MODEL" \
  --output "$OUT" \
  ${LIMIT:+--limit "$LIMIT"} \
  --resume \
  --tensor-parallel-size "${TP_SIZE:-1}"

echo "Done. $OUT written. (Index NOT rebuilt; gemma4 enrichment file untouched.)"
