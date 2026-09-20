#!/usr/bin/env bash
set -euo pipefail

# Simple helper to choose and start Chroma in either persistent (LMDB) or HTTP mode.
# Default: persistent (fastest for local use).
#
# Usage:
#   ./scripts/start_chromadb.sh            # guided, defaults to persistent
#   CHROMA_MODE=http ./scripts/start_chromadb.sh   # force http
#
# Exports CHROMA_MODE and related envs for the current shell session if you source it:
#   source ./scripts/start_chromadb.sh
#

MODE="${CHROMA_MODE:-}"
if [[ -z "$MODE" ]]; then
  echo "Select Chroma mode:"
  echo "  1) persistent (default, local LMDB storage)"
  echo "  2) http        (run chroma server on a port)"
  read -r -p "Choice [1]: " choice
  if [[ -z "$choice" || "$choice" == "1" ]]; then
    MODE="persistent"
  else
    MODE="http"
  fi
fi

export CHROMA_MODE="$MODE"

if [[ "$MODE" == "http" ]]; then
  export CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
  export CHROMA_PORT="${CHROMA_PORT:-8000}"
  echo "➡️  Starting Chroma HTTP server at http://${CHROMA_HOST}:${CHROMA_PORT}"
  echo "    (CTRL+C to stop; ensure 'chroma' CLI is installed)"
  # Run server in foreground so user can stop it; they can nohup if desired.
  chroma run --host "$CHROMA_HOST" --port "$CHROMA_PORT"
else
  export CHROMA_PATH="${CHROMA_PATH:-data/chroma_db}"
  mkdir -p "$CHROMA_PATH"
  echo "SUCCESS: Chroma persistent mode selected."
  echo "   Using path: $CHROMA_PATH"
  echo "   No server to start; embeddings will be stored locally."
fi
