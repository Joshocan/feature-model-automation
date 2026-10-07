#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "${REPO_ROOT}/scripts/install_requirements.sh"
echo "Next: ./.venv-release/bin/python -m pytest -q"
echo "See docs/reproduce.md for data restoration. No generation has been started."
