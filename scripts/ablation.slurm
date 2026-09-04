#!/usr/bin/env bash
#SBATCH --job-name=ogd-ablate
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
cd "${REPO:-$HOME/projects/open-govt-data}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_USE_FLASHINFER_SAMPLER=0
source .venv/bin/activate
python -u -m bench.ablation_check --tier "${TIER:-1}" ${LIMITQ:+--limit-questions "$LIMITQ"} --model "${MODEL:-Qwen/Qwen3-32B-AWQ}"
