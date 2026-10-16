# Generated-side best-match audit

This is a post-hoc, own-reference, names-only audit of semantic correspondence.
It does not validate recall or automatically replace the primary threshold.
No generation, network or embedding calls are made.

## 1. Freeze the sample

From the repository root:

```bash
./.venv-release/bin/python scripts/prepare_pair_audit.py \
  --pairs results/ifs-2027/analysis/semantic-current-v3/pairs.csv \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --seed 20261013 --per-band 20 \
  --output results/ifs-2027/analysis/pair-audit-v1
```

The sampler validates complete per-run pair matrices and the inventory linkage.
For each generated node occurrence it selects the highest similarity reference
occurrence, breaking exact ties by lowest reference index. Best matches below
0.4 are excluded. Twenty occurrences are sampled without replacement in each
band: [0.4,0.5), [0.5,0.6), [0.6,0.7), [0.7,0.8), [0.8,1]. Smaller pools are fully
sampled and the shortage reported; they are never padded or replaced from other
bands. Repeated name pairs across runs/occurrences remain eligible. This targets
generated occurrences, not distinct conceptual pairs. The sample is pooled across
corpora and models, not balanced across them.

Review the frozen protocol before distributing anything. Do not resample after
seeing labels. Preserve the original templates unchanged; raters should **save
completed copies under different filenames**. The analysis verifies frozen hashes.

## 2. Distribute only blinded files

Give rater 1 `ratings_R01.csv`, rater 2 `ratings_R02.csv`, and both raters
`RATER_INSTRUCTIONS.md`. Do not send the whole audit folder or `AUTHOR_ONLY/`.
The latter contains the sample key, population counts, seed and hash manifest.

Raters see item ID, domain and two feature names only. They fill `label` using
exactly `same_concept`, `related`, `unrelated` or `cannot_judge`, plus an optional
comment. Definitions are in the instructions. No model, run, similarity or band
is included. Domain and distinctive feature names can still permit inference;
blinding is not a guarantee of anonymity. Researchers who prepared the key should
not inspect it while independently rating. Do not selectively consult hierarchy
or descriptions; use `cannot_judge` when the names are insufficient.

Keep original returned files private, including any comments that may identify
raters. Agree any reconciliation only after both independent ratings are saved.
Do not overwrite independent ratings with consensus labels.

## 3. Analyse the two completed copies

Replace the two placeholder paths with the actual private returned files:

```bash
./.venv-release/bin/python scripts/analyse_pair_audit.py \
  --audit-dir results/ifs-2027/analysis/pair-audit-v1 \
  --rater1 /absolute/path/to/completed_R01.csv \
  --rater2 /absolute/path/to/completed_R02.csv \
  --output results/ifs-2027/analysis/pair-audit-analysis-v1
```

Use a new output directory. Every sampled item must have a valid label; blanks,
duplicate/missing IDs and changed feature text are rejected. Row order may differ.

Outputs:

- `band_summary.csv`: each rater's category counts, sampled/judged denominators,
  same-concept share among all sampled items and among judged items.
- `weighted_summary.csv`: occurrence-population estimates using band sizes.
  Cannot-judge share is retained separately; conditional same-concept share is
  a ratio estimate. Never use an unweighted mean of the five band percentages as
  population correspondence accuracy.
- `agreement.csv`: raw agreement and **unweighted nominal Cohen's kappa** overall
  and per band, excluding items where either rater cannot judge. Undefined kappa
  (no jointly judged items or constant identical marginals) is null, not zero.
- `confusion.csv`: all 16 label combinations, including cannot judge.
- `summary.json`: provenance, exclusions and interpretation limits.

Overall kappa describes the band-balanced sample, not population-weighted
agreement. No confidence intervals or significance tests are produced: runs and
repeated occurrences may be dependent. Do not interpret missing evidence as an
unrelated label. Author judgement and names-only ambiguity remain limitations.
If a new threshold is suggested by this audit, report that as exploratory rather
than silently changing the original analysis.
