#!/usr/bin/env bash
#
# Serve the CBS metadata search UI on a server (headless, configurable port).
#
#   PORT=8080 ./scripts/serve_app.sh
#
# Then reach it via SSH port-forward from your laptop, e.g.:
#   ssh -N -L 8501:localhost:8080 <netid>@<server>
# and open http://localhost:8501
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"
PORT="${PORT:-8501}"

if [ ! -f data/processed/cbs_search.db ]; then
  echo "No search index found. Build it first:  $PYTHON -m cbs.build_search_index" >&2
  exit 1
fi

exec "$PYTHON" -m streamlit run cbs/search_app.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port "$PORT"
