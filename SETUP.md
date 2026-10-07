# Installation and tests

Run from the repository root. Target interpreter: Python 3.12. Do not install
over the historical experiment environment when preparing a release.

```sh
python3.12 -m venv .venv-release
./.venv-release/bin/python -m pip install -r config/requirements-macos-arm64-py312.lock
./.venv-release/bin/python -m pip check
./.venv-release/bin/python -m pytest -q
```

The lock above targets macOS arm64/Python 3.12. Other platforms must resolve
`config/requirements.txt`, run checks and record their own lock; they are not yet
certified by this release. This is a complete version lock, not a wheel-hash lock.
The primary encoder pins transformers 4.57.6 and torch 2.8.0
to match the evaluation contract. Legacy SentenceTransformer-based helpers are
optional and are not the primary semantic evaluator.

Tests use synthetic chunk identities and blank workbook fixtures. Optional local
corpus/FeatureIDE integration checks may skip when their prerequisites are absent.
The test suite does not require API keys or make generation calls.

## Optional generation dependencies

Install `config/requirements-ingestion.txt` only to rebuild corpus chunks. It adds
PDF extraction dependencies and may require platform-specific system tools.
Source-paper access and an Ollama embedding service are separate requirements.
Refer to IFS_2027_EXPERIMENT.md before rebuilding any frozen inputs.

## FeatureIDE and encoder

Follow docs/featureide-evaluation.md for the verified FeatureIDE 3.10.0 JAR and
its checksum. A JDK is required, not Eclipse. For semantic evaluation, acquire
the snapshot named in config/evaluation/ifs-2027-v0.1.0.json and verify its
identity through the evaluator. Downloading dependencies/model weights requires
network access; evaluation after setup uses local artifacts.

## Safety

Keep keys outside Git. Install/test scripts must not start generation or overwrite
saved outputs. Prefer the explicit commands above to changing your existing .venv.
The former SS/IS launchers and Windows bootstrap instructions are retired.
