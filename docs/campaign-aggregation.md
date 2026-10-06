# — cross-run aggregation and criterion divergence

This stage joins the long-form metric CSVs from the structural, semantic
and provenance evaluators into a campaign-wide view, then produces
the three artefacts the paper turns on: a per-cell summary table, a
variability report anchored on the **degenerate-share headline**, and a
criterion-divergence rank table.

```bash
./.venv/bin/python scripts/aggregate_campaign.py \
  --structural results/ifs-2027/analysis/structure-2026-09-26-v1 \
  --semantic   results/ifs-2027/analysis/semantic-2026-09-26-v1 \
  --provenance results/ifs-2027/analysis/provenance-2026-09-27-v1 \
  --output     results/ifs-2027/analysis/campaign-2026-09-27-v1
```

The output directory must be a fresh subdirectory under
`results/ifs-2027/analysis`. `--semantic` and `--provenance` are optional
(useful early in the pipeline when only structural has landed); the driver
skips whichever streams are absent and records that in `summary.json`.

## Outputs

* `wide.csv` — one row per run, one column per `{source}__{metric}` plus a
  parallel `{source}__{metric}__status` column. The envelope contract from
  Phases 2–4 is preserved so the join never silently zero-fills.
* `cell_summary.csv` — one row per campaign cell (default keys:
  corpus × model × arm × N × grounding × metamodel_block × ordering_id ×
  k_doc). For every numeric metric the summary carries `mean`, `median`,
  `sd`, `min`, `max`, `n_ok`, `n_total`, and one `n_<status>` count per
  non-ok state.
* `variability.csv` — degenerate share per cell + SD of shape metrics
  (n_features, max_depth, avg_branching, mandatory_ratio) + mean/SD of
  outcome columns (F1_total, recall_reach, structural_conformance rate,
  satisfiable rate).
* `rankings.json` — ordered lists of cells per criterion + scores + ranks.
* `correlations.csv` — Spearman ρ and Kendall τ for every unordered pair of
  criteria.
* `summary.json` — source paths, hashes, status histogram, versions,
  unresolved decisions.

## Cell keys

Default cell keys:

```text
corpus, model_id, arm, N, grounding, metamodel_block, ordering_id, k_doc
```

Runs sharing every key form one comparable cell. `seed` and `repetition`
are nested observations within a cell — that is why the aggregators
compute mean/SD per cell rather than per run. Override with `--cell-key`
if a smaller cell (e.g. drop `arm`) is wanted for a specific comparison;
provide the flag multiple times to build the tuple.

## Envelope handling

Non-`ok` cells are excluded from `mean` / `median` / `sd` — a
`not_applicable` recall must not zero out a headline number. Their counts
land in `n_<status>` so the denominator story is intact:

| Column | Interpretation |
| --- | --- |
| `col__mean` | mean over the `ok` cells only |
| `col__n_ok` | number of runs in the cell that scored `ok` |
| `col__n_total` | number of runs in the cell |
| `col__n_ineligible` | runs that were `ineligible` for that metric |
| `col__n_missing_artifact` | runs missing the input artefact |
| `col__n_unsupported` | runs where the metric was unsupported (e.g. W5) |
| `col__n_not_applicable` | runs where the metric was vacuous (e.g. L2 at N=1) |
| `col__n_evaluator_error` | evaluator raised — flag for follow-up |

Booleans (`structural_conformance`, `satisfiable`, `degenerate`, `is_recency_biased`)
are aggregated as their true-share fraction; the field name is kept as
`__mean` rather than `__true_share` so downstream tables use one uniform
column name across numeric and boolean metrics.

## Variability report

The **degenerate share** per cell is the headline: it is the single number
that tells you "how often did this configuration produce a labelled tree
that passes structural conformance without ever using `<or>` / `<alt>` /
`mandatory`?" The report also includes SD of shape metrics so a
low-variance degenerate cell can be distinguished from a high-variance
occasional-degenerate cell.

Reference-based group-kind confusion and mandatory-agreement matrices are
computed by [`fame.evaluation.variability`](../fame/evaluation/variability.py)
and are available for post-hoc analysis; they are **not** rolled into the
campaign summary CSV to keep it wide-but-flat. Pull them separately when
the paper's Repair-primary table is being generated.

## Criterion-divergence rank table

Three criteria by default (override in `DEFAULT_CRITERIA`):

* **structural** — mean of `structural__structural_conformance` (fraction
  of runs whose XSD + W1–W4 tree gate passed).
* **logical**    — mean of `structural__satisfiable`.
* **semantic**   — mean of `semantic__semantic_f1_total`.

For each criterion the driver produces a ranked list of cells (descending
score, ties averaged) and their per-cell score. It then reports Spearman ρ
and Kendall τ between every pair of criteria. This answers the paper's
load-bearing question: **do the criteria disagree, or do they collapse
into one order?**

Rank correlations near 1 mean the three-criteria framing is decorative.
Correlations near 0 or negative mean the criteria genuinely divide the
model × configuration space differently, which is the paper's central
claim.

The correlations are **descriptive**, not tested for significance here —
the statistical-analysis stage pairs them with confidence intervals
under a bootstrap.

## Deliberate limitations

* Rank correlations drop `None` cells rather than imputing a "worst" rank.
  This makes ρ stable under partial coverage (early in the pipeline when
  semantic hasn't been scored on every cell) but means the coefficient
  refers only to cells with all criteria measurable.
* The variability report is reference-free by default. Reference-based
  measures (group-kind confusion, mandatory agreement) are available
  through the primitives but not stapled onto the summary — pulling them
  in requires a per-corpus reference model and a matched-feature list,
  which the statistical-analysis stage supplies.
* Aggregation trusts the frozen inventory identity: sources are joined on
  `run_id` alone. A run whose metrics.csv rows disagree on its cell keys
  will silently take the first-observed value; a duplicate `(run_id,
  metric)` pair inside one source raises `ValueError` to catch double
  writes.

## Verification

```bash
./.venv/bin/python -m pytest \
  tests/test_campaign_aggregate.py \
  tests/test_criterion_divergence.py -q
```

Fixtures cover: numeric coercion of CSV strings, wide-join across sources,
duplicate-metric rejection, mean/SD dropping non-ok rows, bool-as-fraction
aggregation, per-cell partitioning, degenerate-share headline, SD of shape
columns, Spearman/Kendall on perfect agreement, perfect disagreement, and
tied ranks, `None`-pair dropping, and pairwise correlation output shape.
