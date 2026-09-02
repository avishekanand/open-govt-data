#!/usr/bin/env bash
#SBATCH --job-name=ogd-enrich
#SBATCH --requeue
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# Unified CBS StatLine + Eurostat enrichment (enrich.run_vllm).
#
# One model, one schema, one output file; records carry `publisher`. Resumable:
# re-running skips (publisher, code) pairs already in the output, so a preempted
# job continues where it stopped. Nothing git-tracked is overwritten.
#
#   sbatch scripts/enrich_unified.slurm
#   MODEL=Qwen/Qwen3-32B-AWQ SOURCE=eurostat sbatch scripts/enrich_unified.slurm
set -euo pipefail

REPO="${REPO:-$HOME/projects/open-govt-data}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
SOURCE="${SOURCE:-both}"
OUT="${OUT:-}"
LIMIT="${LIMIT:-}"          # max datasets PER SOURCE (smoke tests)

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
# No CUDA toolkit on this cluster, so FlashInfer cannot JIT its sampling kernel.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$REPO"
source .venv/bin/activate

echo "=== $(date) | model=$MODEL source=$SOURCE gpu=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m enrich.run_vllm \
  --source "$SOURCE" \
  --model "$MODEL" \
  ${OUT:+--output "$OUT"} \
  ${LIMIT:+--limit "$LIMIT"} \
  --resume \
  --gpu-mem-util "${GPU_MEM_UTIL:-0.90}" \
  --tensor-parallel-size "${TP_SIZE:-1}"

echo "=== $(date) | done ==="
