#!/usr/bin/env bash
#
# End-to-end CBS "Dutch Public Data Intelligence Engine" pipeline.
# Idempotent and resumable — safe to re-run; each step skips work already done.
#
# Runs anywhere (laptop or server). Configure via env vars:
#   LIMIT        number of active tables to ingest/enrich   (default 700)
#   MODEL        Ollama model for enrichment                (default gemma4:latest)
#   OLLAMA_HOST  Ollama API endpoint                        (default http://localhost:11434)
#   PYTHON       python interpreter / venv python           (default ./.venv/bin/python)
#   SKIP_ENRICH  set to 1 to skip the (slow) LLM enrichment (default unset)
#
# Usage:
#   ./scripts/run_pipeline.sh
#   LIMIT=200 MODEL=gemma3:latest ./scripts/run_pipeline.sh
#   OLLAMA_HOST=http://gpu-node:11434 ./scripts/run_pipeline.sh
#
set -euo pipefail

# Resolve repo root (this script lives in scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIMIT="${LIMIT:-700}"
MODEL="${MODEL:-gemma4:latest}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

log "Repo: $ROOT"
log "Python: $PYTHON ($($PYTHON --version 2>&1))"
log "LIMIT=$LIMIT  MODEL=$MODEL  OLLAMA_HOST=$OLLAMA_HOST"

log "[1/5] Fetch CBS table catalogue"
$PYTHON -m cbs.catalog

log "[2/5] Batch-ingest semantic metadata for $LIMIT active tables (resumable)"
$PYTHON -m cbs.batch_ingest_statline --limit "$LIMIT" --sample-data 5

if [ "${SKIP_ENRICH:-}" = "1" ]; then
  log "[3/5] SKIP_ENRICH=1 — skipping LLM enrichment"
else
  log "[3/5] doc2query enrichment via Ollama ($MODEL) — resumable, ~35s/table"
  # Verify the Ollama endpoint is reachable before a long run.
  if ! curl -fsS -m 5 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    echo "WARNING: Ollama not reachable at $OLLAMA_HOST — skipping enrichment." >&2
  else
    $PYTHON -m cbs.enrich_cbs --resume --limit "$LIMIT" --model "$MODEL"
  fi
fi

log "[4/5] Build SQLite FTS5 search index"
$PYTHON -m cbs.build_search_index --selftest

log "[5/5] Done. Launch the UI with:"
echo "      $PYTHON -m streamlit run cbs/search_app.py --server.port \${PORT:-8501}"
