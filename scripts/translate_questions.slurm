#!/usr/bin/env bash
#SBATCH --job-name=ogd-translate
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
REPO="${REPO:-$HOME/projects/open-govt-data}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$REPO"; source .venv/bin/activate
echo "=== $(date) | model=$MODEL ==="
python -m enrich.translate_questions --model "$MODEL" \
  --tensor-parallel-size "${TP_SIZE:-1}" --gpu-mem-util "${GPU_MEM_UTIL:-0.90}"
echo "=== $(date) | done ==="
