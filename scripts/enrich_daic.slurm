#!/usr/bin/env bash
#SBATCH --job-name=cbs-enrich
#SBATCH --partition=gpu               # adjust to your DAIC GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# Run the CBS doc2query enrichment + index build on TU Delft DAIC (SLURM).
#
# Ollama runs locally on the allocated compute node and serves gemma4 on the GPU.
# The model (~9.6GB) is cached on the umbrella share so it is pulled only once.
#
# Submit:   sbatch scripts/enrich_daic.slurm
# Tail:     tail -f slurm-cbs-enrich-<jobid>.out
#
set -euo pipefail

# --- paths -------------------------------------------------------------------
REPO="${REPO:-$HOME/projects/open-govt-data}"
# Persist Ollama models on the SIR umbrella share so re-runs skip the download.
export OLLAMA_MODELS="${OLLAMA_MODELS:-/tudelft.net/staff-umbrella/sir/ollama/models}"
export OLLAMA_HOST="127.0.0.1:11434"
MODEL="${MODEL:-gemma4:latest}"
LIMIT="${LIMIT:-700}"

cd "$REPO"
mkdir -p "$OLLAMA_MODELS"

# --- environment -------------------------------------------------------------
# DAIC: load a Python (or your conda env). Adjust to the cluster's module names.
module load miniconda3 2>/dev/null || module load python 2>/dev/null || true
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# --- start Ollama on the compute node ---------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama into \$HOME (no root needed)..."
  curl -fsSL https://ollama.com/install.sh | sh || {
    echo "Ollama install failed — install manually and re-submit." >&2; exit 1; }
fi

echo "Starting ollama serve (GPU: ${CUDA_VISIBLE_DEVICES:-?})..."
ollama serve > "ollama-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# Wait for the server, then pull the model (cached on umbrella after first run).
for i in $(seq 1 60); do
  curl -fsS -m 2 "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
ollama pull "$MODEL"

# --- run the pipeline --------------------------------------------------------
echo "Running enrichment + index build (LIMIT=$LIMIT MODEL=$MODEL)..."
export OLLAMA_HOST="http://127.0.0.1:11434"
SKIP_ENRICH=0 LIMIT="$LIMIT" MODEL="$MODEL" PYTHON="$REPO/.venv/bin/python" \
  bash scripts/run_pipeline.sh

echo "Done. Enriched JSONL + cbs_search.db are under data/processed/."
