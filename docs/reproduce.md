# Reproducing the archived study

## 1. Obtain and verify the data

The data DOI/version is not yet assigned. Obtain the matching non-expert archive
from the study authors until the final release metadata identifies its Zenodo
record. Do not invent or substitute a DOI. Verify bundle SHA256SUMS before
extracting all five bundles into one NEW empty directory. Never extract into an
active experiment checkout. Follow the archive README for bundle verification.

## 2. Restore original paths

Use a fresh software checkout. The archive's artifact-manifest.json records both
the archive path and original software-relative path of every evidence file.

```sh
./.venv-release/bin/python scripts/restore_research_data.py \
  --archive /absolute/path/to/extracted-data \
  --destination /absolute/path/to/fresh-software-checkout
```

This verifies checksums and reports absent data/results files without copying.
Historical software/configuration copies are verified but not restored over the
selected software release. Supply
`--manifest-sha256` with the trusted release hash when available. Repeat with
`--apply` to copy absent files. Existing identical files are retained; differing
files cause failure before any copying. Unsafe paths and duplicate targets fail.
The tool does not modify historical metadata, generate results or fetch private PDFs.
Restore into a compatible source version; do not overwrite newer configuration
with old protocol files merely to bypass a mismatch.

## 3. Offline evaluation

Read docs/evaluation-runbook.md for the inventory → structure/semantic/provenance
→ aggregate sequence. Always choose fresh output directories and pass the same
new inventory to all evaluators. Match the pinned encoder identity first.
Then run matching sensitivity, declared statistical families and sibling agreement
as described in docs/analysis-followup-plan.md and docs/statistical-analysis.md.
FeatureIDE and SAT checks have their own prerequisite checks; unsupported is not
false and missing is not zero.

Historical runbook output suffixes are examples, not an instruction to overwrite
archived versions. The selected archived analysis includes semantic-current-v3,
provenance-current-v2, aggregate-current-v4, tau-matching-v2 and
matching-comparison-v2. Consult the archive analysis-lineage.json for exact hashes.

## 4. Plot saved metrics without model calls

After restoration into the expected original paths:

```sh
./.venv-release/bin/python scripts/plot_paper_figures.py \
  --analysis results/ifs-2027/analysis \
  --output results/ifs-2027/analysis/paper-figures-reproduced
```

The script selects saved input versions internally. Inspect its input checks and
figure manifest, and compare generated figure-data CSVs with the archived source
CSVs. PDF metadata may differ without numerical differences. Do not claim complete
reproduction from visual similarity alone.

## 5. Record a reproduction result

Record software commit, interpreter/platform, complete dependency lock, data
manifest hash, commands, test results and per-artifact comparison results. Report
skipped checks and discrepancies. Full scientific reproduction is pending until
this record exists; installation and unit tests are only the first stage.

Generation is a separate paid workflow. Source papers, chunks and Chroma indexes
are not all in the public archive. Do not regenerate campaigns to test installation.
