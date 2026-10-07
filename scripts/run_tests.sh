#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-release"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Run scripts/install_requirements.sh first (creates .venv-release)." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m pip check

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" "${VENV_DIR}/bin/python" -m pytest "$@"
