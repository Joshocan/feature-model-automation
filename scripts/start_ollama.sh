#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Python
if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "ERROR: Python not found."
  exit 1
fi

# Ensure project imports work
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# 🔑 Ensure Homebrew paths are visible (Apple Silicon + Intel)
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Sanity check
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama binary not found on PATH."
  echo "Check: which ollama"
  exit 1
fi

# Models (match your notebook)
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
export OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-llama3.1:8b}"

# Optional API key support
if [[ -f "$PROJECT_ROOT/api_keys/ollama_key.txt" ]]; then
  export OLLAMA_API_KEY_FILE="$PROJECT_ROOT/api_keys/ollama_key.txt"
fi

# Use local Ollama for embeddings; LLM can use cloud if API key is provided
export OLLAMA_EMBED_HOST="http://127.0.0.1:11434"
if [[ -n "${OLLAMA_API_KEY_FILE:-}" ]]; then
  export OLLAMA_LLM_HOST="${OLLAMA_LLM_HOST:-https://ollama.com}"
else
  export OLLAMA_LLM_HOST="${OLLAMA_LLM_HOST:-http://127.0.0.1:11434}"
fi

# IMPORTANT: since you use brew services, server is already running
export OLLAMA_MODE=remote
export OLLAMA_HOST="http://127.0.0.1:11434"

echo "Using Python : $PY ($($PY --version))"
echo "Using Ollama : $(which ollama)"
echo "Ollama mode  : $OLLAMA_MODE"

"$PY" -m fame.services.ollama_service
