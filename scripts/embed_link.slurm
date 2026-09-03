#!/usr/bin/env bash
#SBATCH --job-name=ogd-embed
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
cd "${REPO:-$HOME/projects/open-govt-data}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
source .venv/bin/activate
python -m enrich.embed_link --top-k "${TOPK:-10}" --only "${ONLY:-public_aggregate}"
