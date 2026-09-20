#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -r "${REPO_ROOT}/config/requirements.txt"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" "${VENV_DIR}/bin/python" -m pytest "$@"
