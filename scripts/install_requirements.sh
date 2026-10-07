#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-release"
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

if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
  python -m pip install -r "${ROOT_DIR}/config/requirements-macos-arm64-py312.lock"
else
  echo "No verified lock for this platform; resolving direct requirements."
  python -m pip install -r "${ROOT_DIR}/config/requirements.txt"
fi
python -m pip check

python - <<'PY'
import chromadb, jinja2, lxml, requests, six, tiktoken, xmlschema, yaml
print("SUCCESS: Campaign runtime imports verified")
PY

echo "SUCCESS: Requirements installed into ${VENV_DIR}; no campaigns or corpus downloads run."
