# Reproducing the archived study

## 1. Obtain and verify the data

The data record has reserved DOI `10.5281/zenodo.23206866`; a draft DOI may not
resolve publicly until publication. Obtain the matching archive from its released
record or the authors. Follow the archive README, verify any supplied SHA256SUMS,
and extract into a new directory, never an active experiment checkout. The current
public archive uses descriptive names; restoration reconstructs original paths.

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

## 5.5. Expert assessment (paper §§4.5 and 5.5, Tables 4 and 12)

`scripts/expert_analysis.py` reads existing evidence only. It does not edit the
workbooks, rescore feature names, generate models or contact a provider. The
public data archive supplies anonymised response CSVs and selection metadata.
Original returned workbooks remain private and are not needed for public replication.

After restoring the archive, use:

```sh
./.venv-release/bin/python scripts/expert_analysis.py \
  --public-inputs results/expert-study/public-inputs \
  --output results/expert-study/analysis-reproduced
```

This reads `ratings.csv`, `citation_judgements.csv`, `model_selection.csv` and
`citation_items.csv` from the public-input directory and verifies their identities
and coding. It uses the same statistical calculations and published-value checks
as workbook mode. Input and implementation hashes appropriately differ; compare
the numeric analysis CSVs. A new output directory is required.

Workbook-mode inputs for authorised study maintainers, relative to the checkout:

- `results/expert-study/Expert-1.xlsx` and `Expert-2.xlsx`, using the three
  rating, ranking and citation sheets. Outputs identify raters only as
  **Expert A** and **Expert B**, respectively; no comments, creator properties,
  personal names or original absolute paths are exported.
- `results/expert-study/selection-v3/expert_sample_frozen.csv` and its ten
  hash-verified XMLs.
- `results/expert-study/citations-v3/citation_key_AUTHOR_ONLY.csv`.
- Under `results/ifs-2027/analysis/reproduction-2026-10-07/`:
  `aggregate-current-v4/wide.csv`,
  `matching-comparison-v2/matching_comparison.csv`, and
  `sibling-current-v1/sibling_metrics.csv`.
- `results/expert-study/analysis/astra-n1-rag-evaluation-v1.csv` (all ten
  extension runs, not only the two selected outputs).

The dependency lists include `krippendorff==0.8.2`. For an existing release-test
environment that predates this addition, install it once:

```sh
./.venv-release/bin/python -m pip install krippendorff==0.8.2
./.venv-release/bin/python scripts/expert_analysis.py
```

The default destination is `results/expert-study/analysis-final/`. Existing
outputs are never overwritten. To reproduce again, choose a fresh directory:

```sh
./.venv-release/bin/python scripts/expert_analysis.py \
  --output results/expert-study/analysis-final-repeat/
```

The script applies the researcher-authorised Expert B citation mapping:
`OK → Yes`, `Not exactly → Partially`,
`not sure, depends on the context → Cannot judge`, and `No → No`.
`NJ` ratings and `Cannot judge` citations remain missing, never zero. Decisions
are Reject=0, Accept with revision=1 and Accept=2. Ordinal Krippendorff alpha uses
the full 1..5 domain for ratings and 0..2 for decisions/citations. Rankings use
tie-corrected Kendall W and scipy Kendall tau-b, with negated expert ranks for
comparison against increasing automated scores. Mean rank is the arithmetic
mean of the submitted ranks. Q5 uses the negative independent/one-to-one
precision ratio, a shared predictor rather than two distinct policy metrics.

Main outputs:

| File | Evidence |
| --- | --- |
| `expert_assessment_merged.csv` | One row per selected model, automated P/R/F1 under both policies, conformance/SAT/degeneracy, both raters' scores, decisions and ranks |
| `table4.csv`, `table12_percentiles.csv` | Selected-model table and completed guided-RAG within-cell mid-rank percentiles |
| `rating_summary.csv`, `rating_comparisons.csv`, `ordinal_alpha.csv` | Rating means/missingness, paired rating differences and ordinal reliability |
| `decision_agreement.csv`, `ranking_concordance.csv` | Decision agreement and rank concordance |
| `ranking_correlations.csv`, `rating_metric_correlations.csv` | Within-corpus rank–F1 and rating–measure Kendall tau-b, with eligible n |
| `discordant_pairs.csv` | Every pair reversed by the two F1 policies and the policy each rater prefers |
| `degeneracy_xml_audit.csv`, `variability_summary.csv` | XML-derived degeneracy and Q4 by degeneracy |
| `citation_judgements.csv`, `citation_summary.csv`, `citation_agreement.csv` | Normalised citation responses, support by stratum/corpus and inter-rater agreement |
| `ratings.csv` | Normalised response-level ratings, decisions and ranks |
| `sanity_checks.csv`, `input_hashes.csv`, `software_versions.csv` | Published-value checks, neutral input aliases/hashes and computational versions |
| `summary.md` | Definitions, numerical results and interpretation limits |

Checks run before final tables are written. A discrepancy beyond the supplied
rounding precision exits nonzero and writes only a **STOPPED** summary, checks
and input hashes. The two supplied Astra Table 12 percentile triples are checked;
the other percentile rows are computed but cannot be certified against published
values until the full manuscript table is supplied. Rep-28's XML degeneracy is
computed ignoring empty group leaves; its conformance and unavailable SAT result
are not changed. The sample supports descriptive comparisons, not population-wide
claims about model or policy superiority.
