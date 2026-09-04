#!/usr/bin/env bash
#SBATCH --job-name=ogd-surface
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
cd "${REPO:-$HOME/projects/open-govt-data}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_USE_FLASHINFER_SAMPLER=0
source .venv/bin/activate
python -u -m enrich.surface_forms --publisher "${PUBLISHER:-ESTAT}" \
  --model "${MODEL:-Qwen/Qwen3-32B-AWQ}" ${ONLY_CITED:+--only-cited}
