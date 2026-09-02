#!/usr/bin/env bash
#SBATCH --job-name=ogd-pubev
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# Full-text publication evidence extraction (enrich.pub_evidence).
# Chunks every crawled document end to end, extracts data mentions + research
# questions + verbatim witness sentences, links mentions to the catalogue.
# Resumable: re-running skips URLs already in the output.
set -euo pipefail

REPO="${REPO:-$HOME/projects/open-govt-data}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
LIMIT="${LIMIT:-}"
SAMPLE="${SAMPLE:-}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export VLLM_USE_FLASHINFER_SAMPLER=0   # no CUDA toolkit on this cluster

cd "$REPO"
source .venv/bin/activate

echo "=== $(date) | model=$MODEL gpu=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m enrich.pub_evidence \
  --model "$MODEL" \
  ${LIMIT:+--limit "$LIMIT"} ${SAMPLE:+--sample "$SAMPLE"} \
  --resume \
  --gpu-mem-util "${GPU_MEM_UTIL:-0.90}" \
  --tensor-parallel-size "${TP_SIZE:-1}"

echo "=== $(date) | done ==="
