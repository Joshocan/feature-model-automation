# — statistical hygiene, τ re-thresholding, plotting

The statistical pass turns the aggregated tables from the campaign 5 into the paper's headline
numbers. Everything is CLI-driven and idempotent; the raw run outputs are
never mutated.

The three primitives:

1. **Comparison families** with MW-U + Cliff's δ + Holm correction over a
   *declared* family (no post-hoc family growth).
2. **Percentile bootstrap CIs** with a pinned seed so the same input
   always produces the same interval.
3. **τ re-thresholding** of the raw cosine-pairs table — the paper's τ sweep
   is a single free pass over the raw cosine similarities.

Every helper lives under `fame/evaluation/`; every CLI lives under
`scripts/` and reads/writes ordinary CSVs.

## 1. Declared comparison family

Declare comparisons in a JSON file before scoring. Example
`config/analysis/families/h1b_headline.json`:

```json
{
  "family": "H1b_headline_N10",
  "min_n": 5,
  "alpha": 0.05,
  "comparisons": [
    {
      "name": "H1b_repair_N10_rag_vs_nonrag",
      "arm_a": {"corpus": "repair", "N": 10, "grounding": "rag",    "arm": "headline"},
      "arm_b": {"corpus": "repair", "N": 10, "grounding": "nonrag", "arm": "headline"},
      "metric": "semantic__semantic_f1_total"
    },
    {
      "name": "H1b_federation_N10_rag_vs_nonrag",
      "arm_a": {"corpus": "federation", "N": 10, "grounding": "rag",    "arm": "headline"},
      "arm_b": {"corpus": "federation", "N": 10, "grounding": "nonrag", "arm": "headline"},
      "metric": "semantic__semantic_f1_total"
    }
  ]
}
```

Run it:

```bash
./.venv/bin/python scripts/analyse_family.py \
  --wide       results/ifs-2027/analysis/campaign-2026-09-27-v1/wide.csv \
  --family     config/analysis/families/h1b_headline.json \
  --output     results/ifs-2027/analysis/family-h1b-2026-09-27-v1
```

Outputs under the output dir:

* `family_results.csv` — one row per comparison with `n_a`, `n_b`,
  `mean_a`, `mean_b`, raw `p_value`, Holm-adjusted `p_adjusted`,
  `cliffs_delta`, `delta_magnitude`, `survives_holm`, `borderline`,
  `status` (`ok` / `insufficient_n` / `empty_arm`), and `reason`.
* `borderline_nonsurvivors.csv` — subset that was nominally significant
  (raw p < α) but did **not** survive Holm; these are reported as
  "did not survive", not dropped.
* `bootstrap_intervals.csv` — per-arm mean + 95% percentile-bootstrap
  CI. Skip with `--no-bootstrap` when only significance testing is
  needed.
* `summary.json` — family metadata, family-size accounting
  (`n_significant_pre_holm`, `n_significant_post_holm`,
  `n_borderline_nonsurvivors`), hashes, versions.

### Minimum-n gate

Comparisons where either arm has fewer than `min_n` (default 5)
observations are marked `status="insufficient_n"`: they carry no p-value,
adjusted p or significance verdict. Their slots remain in the declared Holm
family internally as p=1, preserving the declared multiplicity. Mean intervals
also require min_n in the family driver; smaller cells remain descriptive.

The current concrete declarations and commands are in
[analysis-followup-plan.md](analysis-followup-plan.md). They are post-generation
decisions, not preregistration.

### Cliff's δ

Populated on every `ok` comparison. Magnitudes use the
Romano/Kromrey thresholds (`negligible` < 0.147, `small` < 0.33,
`medium` < 0.474, `large` above).

## 2. Bootstrap CIs

`fame.evaluation.bootstrap.mean_ci(values, ...)` returns a pinned-seed
percentile CI or `None` when `n < 2`. Callers filter to `status="ok"`
first (the campaign aggregator's column names help here).

Direct use in analysis notebooks:

```python
from fame.evaluation.bootstrap import mean_ci
values = [row["semantic__semantic_f1_total"] for row in cell_ok_rows]
ci = mean_ci(values, n_resamples=10000)
print(ci.point, ci.lower, ci.upper)
```

The default seed (`20260927`) makes the paper table exactly reproducible
run-to-run. Override with `--bootstrap-resamples` from `analyse_family.py`.

## 3. τ re-thresholding

Rescore every raw cosine pair at any τ **without** re-embedding. Reads
`pairs.csv`, produces `tau_sweep.csv`:

```bash
./.venv/bin/python scripts/tau_rescore.py \
  --pairs   results/ifs-2027/analysis/semantic-2026-09-26-v1/pairs.csv \
  --output  results/ifs-2027/analysis/tau-sweep-2026-09-27-v1
```

Defaults to τ ∈ {0.3, 0.4, 0.5, 0.6} and `independent_max` matching.
Override:

```bash
./.venv/bin/python scripts/tau_rescore.py \
  --pairs   ... \
  --output  ... \
  --tau     0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
  --policy  independent_max one_to_one
```

Output columns per (run × τ × policy): `n_generated`, `n_reference`,
`n_generated_matched`, `n_reference_matched`, `n_matched_pairs`,
`precision`, `recall_total`, `f1_total`, `matched_reference_ids`
(JSON list).

Feed `tau_sweep.csv` back through `analyse_family.py` with a
τ-comparison family declaration to check whether Holm-survivors flip
under different thresholds — the D04 sensitivity story lands here.

## 4. Plotting

### Correlation eligibility

Criterion correlations use only cells with finite observed scores for both
criteria. Spearman and Kendall use exactly the same pairs; missing scores
are never ranked as low performance in these calculations. `correlations.csv`
reports `n`, `n_excluded`, and `status` (`ok`, `insufficient_n`, or
`constant_input`). Coefficients are null when fewer than two pairs remain or
one criterion is constant. This is a computability rule, not a minimum sample
size for an inferential claim. Previously generated correlations must be
regenerated with the corrected aggregator.

`scripts/plot_figures.py` is deliberately CLI-only. All five plots share
the same 6×4 inch default figure at 150 dpi, PDF + PNG side-by-side.
Missing input files produce a labelled placeholder rather than a crash so
the plot directory always documents which figure came from where.

Dry-run first to see what the invocation would produce without touching
matplotlib:

```bash
./.venv/bin/python scripts/plot_figures.py \
  --campaign results/ifs-2027/analysis/campaign-2026-09-27-v1 \
  --tau      results/ifs-2027/analysis/tau-sweep-2026-09-27-v1 \
  --family   results/ifs-2027/analysis/family-h1b-2026-09-27-v1 \
  --output   results/ifs-2027/analysis/plots-2026-09-27-v1 \
  --plot degenerate_share n_curve criterion_heatmap tau_sweep \
  --dry-run
```

Same command without `--dry-run` renders the plots:

| slug | source | figure |
| --- | --- | --- |
| `degenerate_share` | `variability.csv` | horizontal bar chart per cell |
| `n_curve` | `cell_summary.csv` | mean F1_total vs N, one line per (corpus, model, grounding) |
| `criterion_heatmap` | `correlations.csv` | Spearman ρ heatmap between the three criteria |
| `tau_sweep` | `tau_sweep.csv` | F1_total vs τ, one line per run |
| `family_boxplot` | `family_results.csv` + `--wide` | box plot of each comparison's metric |

Plot data prep is factored into pure functions (`prepare_degenerate_share`,
`prepare_n_curve`, `prepare_criterion_heatmap`, `prepare_tau_sweep`) so
extension is trivial and the numeric side stays testable.

## Deliberate limitations

* **The comparison family is fixed at declaration time.** Adding an
  unplanned comparison after seeing outputs violates §6 and D04. Extend
  the JSON declaration and re-run rather than editing results.
* **Bootstrap CIs are percentile method, not BCa.** BCa's acceleration
  correction requires the influence function; percentile is what §6
  explicitly names and is what the paper table uses.
* **τ rescoring never re-embeds.** The similarity column comes verbatim
  from `pairs.csv`; if the encoder changed, run the semantic evaluator again first.
* **Pairing is not verified.** `paired_difference_ci` computes on
  aligned position pairs; a justified pairing design is the caller's
  responsibility.
* **Plots are figures, not tables.** Every headline number must also
  land in the CSV files — the paper cannot cite a figure alone.

## Verification

```bash
./.venv/bin/python -m pytest \
  tests/test_bootstrap.py \
  tests/test_statistical_family.py \
  tests/test_tau_rescore.py -q
```

Fixtures cover: pinned-seed reproducibility, seed drift, insufficient-n
gate, empty-arm reporting, Holm survives + borderline reporting, Cliff's
δ population, τ higher → matches monotonically shrink, τ-policy
sensitivity (independent-max vs one-to-one), and empty-pair zero
measurement.
