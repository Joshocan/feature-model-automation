# FAME Setup (Automated)

> **⚠️ SUPERSEDED — this file targets the retired 2×2 pipeline setup.**
> Update lands in Phase 5 after the unified N-granularity engine is in. Until then, do not rely on the commands below for the new campaign — see [IFS_2027_EXPERIMENT.md](IFS_2027_EXPERIMENT.md).

---

## Quick start (macOS/Linux)
```bash
./scripts/initial_setup.sh
```

What it does:
- Creates/uses `.venv`
- Installs `config/requirements.txt`
- Downloads NLTK punkt
- Sets default hosts:
  - `OLLAMA_EMBED_HOST=http://127.0.0.1:11434`
  - `OLLAMA_LLM_HOST=https://ollama.com`
- Pre-creates common data/results/logs directories

## Windows
Use PowerShell bootstrap (env activation is not persisted by the script):
```powershell
./scripts/bootstrap.ps1
```

## Manual env activation (macOS/Linux)
```bash
source .venv/bin/activate
```

## If you need to reinstall deps
```bash
./scripts/install_requirements.sh
```

## Environment variables to know
- `FAME_BASE_DIR`  : repo root if unset
- `CHROMA_MODE`    : `persistent` (default) or `http`
- `OLLAMA_EMBED_HOST` : local host for embeddings
- `OLLAMA_LLM_HOST`   : host for LLM generation (cloud/local)
- `OLLAMA_API_KEY` / `OLLAMA_API_KEY_FILE` : if your Ollama host requires auth

## After setup
Run the end-to-end launcher:
```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_fame.py
```

Or run steps individually (see README).
