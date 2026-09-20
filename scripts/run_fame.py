#!/usr/bin/env python
"""One-stop launcher for FAME pipelines (Python version).

Features:
 - Optional preprocessing (ingestion + vectorization)
 - Interactive selection: RAG vs Non‑RAG, then SS/IS variant
 - Uses environment defaults: FAME_BASE_DIR, CHROMA_MODE, OLLAMA_* hosts

Run with the repo venv python:
  PYTHONPATH=$(pwd) .venv/bin/python scripts/run_fame.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_venv_python() -> Path:
    if not VENV_PY.exists():
        die(".venv Python not found. Run scripts/initial_setup.sh first.")
    return VENV_PY


def prompt(msg: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    ans = input(f"{msg} {suffix}: ").strip().lower()
    if not ans:
        return default_yes
    return ans.startswith("y")


def choose(title: str, options: list[str]) -> int:
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    choice = input("Select option: ").strip()
    try:
        idx = int(choice)
        if 1 <= idx <= len(options):
            return idx
    except Exception:
        pass
    die("Invalid selection")


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    env.setdefault("FAME_BASE_DIR", str(REPO_ROOT))
    env.setdefault("CHROMA_MODE", "persistent")
    print(f"\n→ {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    py = ensure_venv_python()

    print("==================== FAME Runner (Python) ====================")
    print(f"Repo          : {REPO_ROOT}")
    print(f"Venv Python   : {py}")
    print(f"FAME_BASE_DIR : {os.getenv('FAME_BASE_DIR', REPO_ROOT)}")
    print(f"CHROMA_MODE   : {os.getenv('CHROMA_MODE', 'persistent')}")
    print(f"OLLAMA_EMBED_HOST : {os.getenv('OLLAMA_EMBED_HOST', 'http://127.0.0.1:11434')}")
    print(f"OLLAMA_LLM_HOST   : {os.getenv('OLLAMA_LLM_HOST', 'https://ollama.com')}")
    print("=============================================================")

    # Preprocessing
    if prompt("Run preprocessing (ingestion + vectorization) now?", default_yes=True):
        run([str(py), "scripts/preprocessing_for_rag.py"])
    else:
        print("⏩ Skipping preprocessing")

    family_idx = choose("Choose pipeline family:", ["RAG (retrieval-guided)", "Non-RAG (context-only)"])

    if family_idx == 1:
        variant_idx = choose("RAG variants:", ["Single-Stage RAG (ss-rgfm)", "Iterative RAG (is-rgfm)"])
        if variant_idx == 1:
            run([str(py), "scripts/run_ss_rag.py", "--interactive"])
        else:
            run([str(py), "scripts/run_is_rag.py", "--interactive"])
    else:
        variant_idx = choose("Non-RAG variants:", ["Single-Stage Non-RAG (ss-nonrag)", "Iterative Non-RAG (is-nonrag)"])
        if variant_idx == 1:
            run([str(py), "scripts/run_ss_nonrag.py", "--interactive"])
        else:
            run([str(py), "scripts/run_is_nonrag.py", "--interactive"])

    print("\nSUCCESS: FAME run completed")


if __name__ == "__main__":
    main()
