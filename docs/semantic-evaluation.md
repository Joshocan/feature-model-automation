# — semantic evaluation and encoder identity

This implements offline semantic scoring of completed final outputs against
the frozen reference models, plus D03 encoder-identity verification. All
planned runs are retained; non-completed runs receive explicit
`status="ineligible"` rows and no matcher is invoked for them.

```bash
./.venv/bin/python scripts/evaluate_semantic.py \
  --inventory results/ifs-2027/analysis/inventory-2026-09-26-v2 \
  --output    results/ifs-2027/analysis/semantic-2026-09-26-v1
```

Choose a fresh output directory. The driver verifies encoder identity against
`config/evaluation/ifs-2027-v0.1.0.json → semantic.recorded_pilot_revision`
before scoring any run. A mismatch or missing snapshot aborts the run: no
alternate encoder is substituted silently.

Outputs land under the chosen directory:

* `metrics.csv` / `metrics.json` — long-form envelope per (run, metric).
* `pairs.csv` — raw D15 generated↔reference cosine similarities. **No
  τ-thresholding is applied**; the τ sweep in the statistical pass re-thresholds this
  file.
* `summary.json` — encoder identity, input hashes, dependency versions,
  status-count histogram, and unresolved decisions (D02/D03/D04).

## Metrics emitted

Each row carries the full envelope: `value`, `status`, `reason`, plus
`numerator`/`denominator` where applicable. Semantic metric names:

| Metric | Definition |
| --- | --- |
| `n_generated`, `n_reference` | feature-node counts (root inclusive) |
| `semantic_precision` | fraction of generated features with a max-cosine ≥ τ against any reference feature (independent-max) |
| `semantic_recall_total` | fraction of reference features hit by some generated feature at ≥ τ |
| `semantic_f1_total` | harmonic mean of the two above; labelled with the `total` denominator so it is never silently paired against `reach` |
| `recall_reach` | primary recall against `reach(T, C)` — features with attributed docs in the corpus (D4 ∩ manifest) |
| `recall_attested`, `recall_organising` | recall against the D21 partition halves |
| `matched_reference_ids` | sorted names of reference features hit at τ_primary (kept for downstream provenance/parent-match analyses) |
| `duplicate_generated_hits` | count of reference nodes matched by more than one generated node — the diagnostic that motivates the one-to-one sensitivity policy |
| `F_t_size`, `F_t_attested_size`, `F_t_organising_size`, `reach_size`, `rho_T_C` | corpus-level calibration constants, repeated per run for join safety |

## Envelope statuses

* `ok` — measured value present.
* `ineligible` — run was not completed, or a required input feature-set is
  empty; matcher is not invoked.
* `missing_artifact` — final XML or reference XML absent.
* `not_applicable` — the target partition (e.g. F_t^att) is empty, so its
  denominator is undefined.
* `unsupported` — corpus has no ground-truth artefacts loaded.
* `evaluator_error` — XML parse or encoder call raised.

Zero is a measured score; it is not substituted for a null status.

## Encoder identity (D03)

`fame/evaluation/local_encoder.verify_encoder_identity()` verifies the
Hugging Face snapshot before any embedding is computed. It:

1. Locates the newest local snapshot for `sentence-transformers/all-mpnet-base-v2`.
2. Compares its revision directory name to the pilot-recorded value
   (`e8c3b32edf5434bc2275fc9bab85f82640a19130`).
3. Hashes every weight file (`model.safetensors`, `pytorch_model.bin`) and
   tokenizer artefact present, and stores the digests in `summary.json`.
4. Optionally verifies package versions (transformers, torch) against the
   recorded pilot versions.

`status="ok"` requires **all** of the above. Any mismatch produces a
non-zero exit with the reason recorded — no fallback encoder is loaded and
no partial similarity file is written.

## τ sweep

The default sweep `{0.3, 0.4, 0.5, 0.6}` from the contract is baked into
`pairs.csv`: every raw similarity is retained, so re-thresholding at any τ
is a one-pass filter. The driver also writes a per-run `tau_sweep_view`
summary into each row's envelope for quick eyeballing; this convenience is
not authoritative — the CSV is.

## Deliberate limitations

* The primary semantic population is completed finals with an extractable
  feature tree (D01, author decision 2026-09-29). The driver records metrics
  for every planned run; downstream analysis uses the explicit primary gate in
  `wide.csv`. Strict-admissible finals are a sensitivity population.
* Parent-match structural alignment and provenance markers are provenance-stage
  concerns and are **not** produced by this driver.
* One-to-one maximum-cardinality sensitivity is available through
  `prf_from_similarity(..., matching_policy="one_to_one")` but the default
  `evaluate_semantic()` reports the independent-max primary. Sensitivity
  scoring is run in a separate analysis pass to keep the D15 pairs
  representation policy-agnostic.
* `rho_T_C` is read from `data/calibration/rho.json` when present; otherwise
  it is derived on the fly from `|reach|/|F_t|` and marked
  `not_applicable` if `F_t` is empty.

## Verification

```bash
./.venv/bin/python -m pytest tests/test_semantic.py -q
```

Fixtures cover envelope missing-artifact / ineligible / not-applicable
handling, dual recall against attested/organising/reach partitions, τ-sweep
consistency, raw-pair completeness, and D03 revision / version / missing-
weight mismatches.

## Historical direct-attribution calibration correction (2026-09-30)

This subsection documents the earlier delimiter repair. Its reach=attested
expectation is superseded by the ancestor-closure definition below.

Both evaluation drivers now use `fame/evaluation/manifests.py` to read the
semicolon-separated manifests. Semantic calibration validates partition coverage,
disjointness, reference identity uniqueness and attribution document membership.
Rho is derived from current reach; the historical calibration JSON is only a
consistency check (absolute tolerance 0.00005 for its four-decimal rounding).
Corpus reach describes annotated corpus availability, not the retrieved passages.

The existing `semantic-current-v1` reach metrics are invalid and must be
regenerated. Run the following sequentially, using fresh output directories:

```bash
./.venv/bin/python scripts/evaluate_semantic.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/semantic-current-v2

./.venv/bin/python scripts/aggregate_campaign.py \
  --structural results/ifs-2027/analysis/structure-current-v1 \
  --semantic results/ifs-2027/analysis/semantic-current-v2 \
  --provenance results/ifs-2027/analysis/provenance-current-v2 \
  --output results/ifs-2027/analysis/aggregate-current-v3
```

This standard semantic command recomputes embeddings locally; it makes no
generation calls. No cached-metric repair command is introduced by this fix.
Expected startup calibration: Repair reach=58, Federation reach=36. For these
corpora, reach recall should equal attested recall. Existing full-reference
scores and raw similarities should remain unchanged under the same encoder.
Keep old outputs for audit. Threshold results do not depend on reach, so a
threshold rerun is not required solely for this correction.

## Definition 2: reference ancestor closure (2026-09-30)

Reach now includes directly supported reference features and all their reference
ancestors, including organising ancestors. Missing parent links and cycles fail
validation. The historical calibration JSON is checked against the direct set,
not used as the new rho. New summaries record the definition version, direct-set
size, added ancestors and complete reach membership. Inventory contract hashes
remain source identifiers; retain this post-generation definition addendum and
implementation hashes with the exports.

Verified against the current reference XMLs: Repair has 58 direct seeds plus 26
ancestors = 84/100 (rho=0.84). Federation has 36 direct seeds plus 23 ancestors =
59/130 (rho=0.45384615384615384). These are calibration checks, not a full rerun.

Use fresh directories to avoid mixing direct-only and closed-reach results:

```bash
./.venv/bin/python scripts/evaluate_semantic.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/semantic-current-v3

./.venv/bin/python scripts/aggregate_campaign.py \
  --structural results/ifs-2027/analysis/structure-current-v1 \
  --semantic results/ifs-2027/analysis/semantic-current-v3 \
  --provenance results/ifs-2027/analysis/provenance-current-v2 \
  --output results/ifs-2027/analysis/aggregate-current-v4
```

This recomputes local embeddings, not model generation. Do not assert reach recall
equals attested recall. For closed reach, the complementary partition in the
weighted recall identity is `F_t minus reach`. Recall above rho still does not
alone establish a lack of supporting evidence.
