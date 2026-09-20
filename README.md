# Feature Model Automation

> **⚠️ SUPERSEDED — this README describes the retired 2×2 pipeline design.**
> The active plan is [IFS_2027_EXPERIMENT.md](IFS_2027_EXPERIMENT.md), which replaces the four SS/IS×RAG/Non-RAG pipelines with a single N-granularity engine. A full rewrite of this README lands in Phase 5 (unified engine). The text below is kept only for orientation while the migration is in flight.

---

This repository contains the feature-model automation experiments formerly developed within FAME. It builds FeatureIDE-compatible feature models from textual artefacts using four LLM pipeline families. The internal Python package remains `fame` to preserve the existing module imports and experiment commands.

- `ss_nonrag` — single-stage prompt-based context conditioning
- `is_nonrag` — iterative prompt-based context conditioning
- `ss_rag` — single-stage retrieval-augmented generation
- `is_rag` — iterative retrieval-augmented generation

For the paper-specific reproduction workflow, see:

- `IFS_2027_EXPERIMENT.md` (iFS @ ETAPS 2027 experiment guide)

## 1) Initial setup

- See `SETUP.md` for full setup details.
- macOS/Linux quick start:

```bash
./scripts/initial_setup.sh
```

For stable runs, keep the Python environment outside OneDrive. A typical setup is:

```bash
python3 -m venv $HOME/.venvs/fame
source $HOME/.venvs/fame/bin/activate
PYTHONNOUSERSITE=1 python -m pip install --upgrade pip
PYTHONNOUSERSITE=1 python -m pip install -r config/requirements.txt
```

All commands below assume:

```bash
source $HOME/.venvs/fame/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH=$(pwd)
```

## 2) End-to-end launcher

Runs optional preprocessing, then lets you pick RAG / Non-RAG and SS / IS variants.

```bash
python scripts/run_fame.py
```

## 3) Run individual steps

### Preprocessing (ingest + vectorize)

```bash
python scripts/preprocessing_for_rag.py
```

### Single-stage Non-RAG

```bash
python scripts/run_ss_nonrag.py --interactive
```

### Iterative Non-RAG

```bash
python scripts/run_is_nonrag.py --interactive
```

### Single-stage RAG

```bash
python scripts/run_ss_rag.py --interactive
```

### Iterative RAG

```bash
python scripts/run_is_rag.py --interactive
```

## 4) Evaluation helpers

Coverage for a single FM:

```bash
python scripts/coverage_fm.py \
  --gt data/ground_truth/federation.xml \
  --pred results/rag/ss-rgfm/fm/your_model.xml
```

Well-formedness / XSD conformance:

```bash
python scripts/check_wellformed.py \
  --xml results/rag/ss-rgfm/fm/your_model.xml \
  --xsd prompts/specifications/feature_model_featureide.xsd
```

Duplicate feature names:

```bash
python scripts/check_feature_duplicates.py \
  --xml results/rag/ss-rgfm/fm/your_model.xml
```

## 5) Outputs and directories

Main output locations:

- generated FMs: `results/**/fm/`
- pipeline reports / metadata: `results/**/reports/`
- top-ranked FMs: `results/**/top_fm/`
- overall analysis datasets: `results/analysis/`
- curated final FMs: `final_fm/`

## 6) Environment notes

- Chroma database defaults to:
  - `data/chroma_db`
- Ollama embedding default:
  - `OLLAMA_EMBED_MODEL=nomic-embed-text`
- evaluation embedding default:
  - `all-mpnet-base-v2`

If the repository is stored under OneDrive, keep at least these outside OneDrive:

- Python virtual environments
- Chroma persistent storage
- temporary runtime / analysis copies

For more setup detail, read `SETUP.md`.
