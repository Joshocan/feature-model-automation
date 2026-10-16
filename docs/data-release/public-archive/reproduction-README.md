# Reproduction evidence and commands

`automated-evaluation/` preserves the recorded clean-environment evaluation:
17 command stages succeeded and 42 CSV comparisons were byte-identical.
Execution dates remain in the records, not folder names. `commands.json`
contains the exact historical commands, `environment.json` the recorded
dependencies/source hashes, `comparison.json` the comparisons and `logs/` the
captured output. Historical absolute paths identify the original execution;
replace the checkout prefix when running locally.

Identical reproduced data are not stored twice. The archive manifest records
`restore_sources` aliases for those files and copies distinct reproduction
artifacts under this directory. Restoration reconstructs every recorded
software-relative path. This historical check does not certify later source
changes or a future release tag.

`expert-public-input-check.json` records the separate check that anonymised CSV
inputs reproduce the expert assessment without private workbooks. The test
compares all numeric analysis tables against the original analysis, excluding
input/source-version hashes. Packaging and source-code changes still require
final CI and a compatible release identifier.

## Offline commands after restoration

From the updated software checkout, select new output directories:

```sh
# Expert assessment, including Tables 4 and 12.
./.venv-release/bin/python scripts/expert_analysis.py \
  --public-inputs results/expert-study/public-inputs \
  --output results/expert-study/analysis-reproduced

# Paired negative-control F1 intervals from saved run-level scores.
./.venv-release/bin/python scripts/bootstrap_cross_reference.py

# Largest Federation own-minus-cross mean over both policies and six thresholds.
./.venv-release/bin/python scripts/bootstrap_cross_reference.py --federation-maximum

# Attribution, reference-partition and logical-validity checks; reuse search evidence.
./.venv-release/bin/python scripts/reanalyse_provenance_followup.py \
  --reuse-search results/reanalysis_2026-10-15-final \
  --output results/verification-reproduced \
  --report results/verification-reproduced.md

# Publication figures from saved metrics.
./.venv-release/bin/python scripts/plot_paper_figures.py \
  --analysis results/ifs-2027/analysis \
  --output results/ifs-2027/analysis/figures-reproduced
```

The legacy script and restored source paths intentionally retain their original
names; archive folder names are descriptive. Never rename inputs inside a live
software checkout by hand. The restore tool performs the mapping.

For complete XML-to-metric reproduction use `scripts/validate_reproduction.py`
and the companion reproduction guide. That requires the pinned local embedding
model and Java/FeatureIDE dependencies. Check its comparison report, not only
the process exit code: the historical driver does not fail on every comparison
discrepancy. New evaluation runs are not necessary to inspect the saved tables.

Final compatible software commit/tag and human manuscript sign-off remain in
`RELEASE_APPROVAL.md`. A previous successful run is not a substitute for validating
the final frozen software/data pair.
