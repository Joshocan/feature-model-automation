# FeatureIDE batch compatibility evaluation

This is an additional post-generation compatibility audit, not a change to the
frozen scoring population. It uses FeatureIDE's actual `XmlFeatureModelFormat`
reader, via a small Java bridge driven by Python. No Eclipse, model-generation
calls, embeddings, SAT solving, or changes to saved XML are involved.

## Pinned dependency

FeatureIDE 3.10.0 is a fixed compatibility target, not a claim to use the newest
release. Official download listing: https://featureide.github.io/ .
The standalone library was verified locally with Java/Javac 8; it suffices for
this reader without additional JARs. A JDK (`java` and `javac`) is required.
The adapter registers the default model factory explicitly outside Eclipse.

The library has already been downloaded in this workspace. For another checkout:

```bash
mkdir -p tools/featureide/lib
curl -L --fail \
  -o tools/featureide/lib/de.ovgu.featureide.lib.fm-v3.10.0.jar \
  https://github.com/FeatureIDE/FeatureIDE/releases/download/v3.10.0/de.ovgu.featureide.lib.fm-v3.10.0.jar
```

Expected SHA-256 (enforced by the driver):
`b718c2461b4af2eccd64c5afd06ac804554bd723d2685d4521da423c9ef1829a`.
JARs and compiled classes are ignored by git. Retain this URL, hash and upstream
licensing information in replication materials; review upstream license obligations
before redistributing binaries. Java classes compile into a temporary directory.

## Run from the repository root

Optional preflight only (no campaign outputs processed):

```bash
./.venv/bin/python scripts/evaluate_featureide.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/featureide-preflight-v1 \
  --preflight-only
```

Full batch, including the same preflight:

```bash
./.venv/bin/python scripts/evaluate_featureide.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/featureide-current-v1
```

This selects `completed=True` from `runs.csv` (currently 500 of 694) and reads
`fm_gen.xml` under each `resolved_path`. It does not select only XSD-valid models.
Each model runs in an isolated JVM, with a 512 MB heap and 30-second timeout;
use `--timeout` to declare another limit. Execution is sequential and local.
Do not regenerate source artifacts concurrently. Use a fresh output directory
for every rerun. The full run is deliberately not launched during implementation.

## Results and interpretation

- `featureide_parse.csv`: one row per completed run, `run_id`, configuration,
  XML path/hash, `structural__featureide_parse`, status/reason, error/warning
  counts, loaded feature count, detailed parser problems and stderr.
- `summary.json`: inventory, JAR, implementation and result hashes, Java version,
  control results, population size and acceptance/rejection/error totals.

`status=ok, parsed=False` is a reader rejection; `status=ok, parsed=True` means
no reader errors and a loaded root. Warnings are preserved but do not cause
rejection. Missing files, timeouts and runtime failures leave the value null,
not false. DTD/entity inputs are refused by the adapter safety policy and recorded
as evaluator errors, not FeatureIDE rejections. No external resources are loaded.
Exit code 1 signals unavailable evaluations, not ordinary model rejection.

Preflight must accept a known valid tree and reject malformed XML and duplicate
names. Both saved reference models are also tested and recorded. Reference
rejection is a diagnostic result, not automatically a broken evaluator.

The CSV is joinable on `run_id`; the 194 incomplete runs remain ineligible.
Existing aggregate files are not overwritten, and strict-admissibility flags
are not changed. Do not rerun the old aggregation expecting it to ingest this
new CSV automatically. Decide/document the integration rule first, then build
a new versioned aggregate. Report this as **FeatureIDE 3.10.0 XML-reader
acceptance**, not Eclipse UI compatibility, semantic correctness, satisfiability,
or evidence that all input metadata survived import. Compare `loaded_features`
with source feature counts when investigating potential import transformations.

## Tests

```bash
./.venv/bin/python -m pytest -q tests/test_featureide.py
```

The integration test uses the real pinned JAR and both references; it skips when
the JAR or compiler is absent. Unit tests distinguish rejection from unavailable
evaluation and check completed-run selection and duplicate-ID protection.
