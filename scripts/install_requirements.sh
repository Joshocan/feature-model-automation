#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
CAMPAIGN_PYTHON="${CAMPAIGN_PYTHON:-}"

if [[ -z "${CAMPAIGN_PYTHON}" ]]; then
  if [[ -x "/usr/local/bin/python3.12" ]]; then
    CAMPAIGN_PYTHON="/usr/local/bin/python3.12"
  else
    CAMPAIGN_PYTHON="$(command -v python3)"
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${CAMPAIGN_PYTHON}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/config/requirements.txt"

python - <<'PY'
import chromadb, jinja2, lxml, requests, six, tiktoken, xmlschema, yaml
print("SUCCESS: Campaign runtime imports verified")
PY

python - <<'PY'
import nltk
nltk.download("punkt", quiet=True)
PY

echo "SUCCESS: Requirements installed into ${VENV_DIR} and NLTK punkt downloaded."
