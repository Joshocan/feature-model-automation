#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/config/requirements.txt"

python - <<'PY'
import nltk
nltk.download("punkt", quiet=True)
PY

echo "SUCCESS: Requirements installed into ${VENV_DIR} and NLTK punkt downloaded."
