# — provenance evaluation (L0, L1, L2, recency)

This is the offline provenance pass: given a completed run's `fm_gen.xml`,
its `fm_iter/step_*.xml` checkpoints and its `context_log.jsonl`, emit
envelope-form metrics for every level of the provenance ladder and the
recency-bias sanity check that keeps them interpretable.

```bash
./.venv/bin/python scripts/evaluate_provenance.py \
  --inventory results/ifs-2027/analysis/inventory-2026-09-26-v2 \
  --output    results/ifs-2027/analysis/provenance-2026-09-27-v1
```

The driver refuses to run when the inventory has issues, when the output
directory already exists, or when a final artefact's sha256 differs from
what the inventory recorded — no silent recovery, no artefact rewrite.

## Outputs

* `metrics.csv` / `metrics.json` — long-form envelope per (run, metric)
  with `value`, `status`, `reason`, and numerator/denominator where
  applicable.
* `citations.csv` — D16 rows: one per parseable `(feature_id, cited_doc_id,
  first_seen_step)` triple across all runs.
* `hallucinated.csv` — L2 offenders: citations whose doc_id was known to
  the corpus but not in the context at the birth step.
* `summary.json` — encoder-agnostic; carries hashes, package versions,
  status histogram, and outstanding decisions.

## Metrics emitted

| Metric | Definition |
| --- | --- |
| `n_features` | feature-node count (root inclusive) |
| `n_features_with_description` / `n_features_with_trace` / `n_features_with_parseable_trace` | staircase from "wrote something" to "parsed by the frozen grammar" |
| `feature_trace_coverage` | fraction of features carrying a parseable trace |
| `L0_marker_emission_compliance` | fraction of features emitting a marker (regardless of parse) |
| `L0_parse_rate` | of emitted markers, the fraction that parse — `not_applicable` when no marker was emitted |
| `n_citation_pairs` / `n_unique_cited_doc_ids` | citation-pair-level counts |
| `L1_referential_integrity` | fraction of cited doc_ids that exist in the D1 manifest — `unsupported` when no known set is supplied, `not_applicable` when there are no citation pairs |
| `L2_hallucination_rate` | rate of citations whose doc was **not** in the context at `first_seen_step`; `not_applicable` for Non-RAG N=1 (vacuous), `missing_artifact` when the context log is absent |
| `L2_n_checked` | denominator behind the rate |
| `L2_applicable` | boolean — false when N=1 makes the check vacuous |
| `recency_mean_offset` | mean of `citation_step − doc_first_seen_step` |
| `recency_zero_offset_share` | fraction of citations at offset 0 — the recency-bias signature |
| `recency_max_offset` | worst-case offset, used to distinguish "no spread available" from "spread available, model ignored it" |
| `recency_negative_offset_count` | bookkeeping-integrity check; should always be 0 |
| `is_recency_biased` | boolean heuristic — `zero_offset_share > 0.5 AND max_offset > 0` |

## Envelope statuses

* `ok` — measured value present.
* `ineligible` — run was not completed, or `fm_gen.xml` has no feature nodes;
  no metric is invoked downstream.
* `missing_artifact` — final XML, iterate directory, or context log absent
  when needed.
* `not_applicable` — L2 vacuous under Non-RAG N=1; recency undefined when
  no citations resolve to a batch; L0 parse rate undefined when no marker
  was emitted.
* `unsupported` — L1 without a known-doc-id set.
* `evaluator_error` — XML parse or marker-grammar failure.

Zero is a measured score; it never substitutes for an unknown status.

## L2 vacuity rule (Non-RAG N=1)

When `N=1` **and** the single step's `batch_doc_ids ∪ chunk_doc_ids`
covers every cited doc, every valid citation is trivially in context
and L2 gives no evidence of grounding. The driver marks the metric
`not_applicable` rather than presenting a spurious `0.0`. This follows
the IFS brief §6.6 policy: "vacuous, not zero."

## Marker grammar

The frozen grammar (`fame/utils/marker_grammar.py`) requires each
`<description>` to end with exactly:

```
Trace: [id1, id2]
```

Any deviation — legacy `[src: …]` markers, trailing text after `]`, missing
`Trace:` prefix, or empty bracket — fails parseably and lands in the L0
`marker_emission_compliance` numerator but not `L0_parse_rate`. This
staircase is preserved so downstream analysis can distinguish "no attempt"
from "grammatical mistake."

## first_seen_step derivation

The driver walks `fm_iter/step_00.xml` … `step_(N-1).xml` in order, parses
every description, and records the earliest step index at which each
`(feature, doc_id)` pair appears. A later step re-emitting the same pair
does **not** overwrite the earlier record. This is the anchor for both L2
and recency: L2 asks "was the doc in context at that step?"; recency asks
"how does that step relate to when the doc first entered any batch?"

## Deliberate limitations

* L2 uses `batch_doc_ids ∪ chunk_doc_ids` as the definition of "in
  context". A doc that appeared only in retrieved chunks but not in the
  current batch is still counted as in-context. This matches the pilot's
  policy; separating the two sources is deferred to a diagnostic layer.
* Recency is a per-run signal only. Cross-run aggregation happens in
  cross-run aggregation where run-level `is_recency_biased` and `zero_offset_share`
  feed the fleet report.
* `L1_referential_integrity` uses the D1 manifest doc_ids. It cannot
  audit passage-level support for a citation — that is D05 (human
  validation).
* The evaluator does **not** compare the marker text against the
  reference model. Cross-referencing citations against ground-truth
  attribution (D4) is a cross-run concern.

## Verification

```bash
./.venv/bin/python -m pytest tests/test_provenance.py \
  tests/test_marker_grammar.py \
  tests/test_hallucination.py \
  tests/test_recency.py -q
```

Fixtures cover: missing gen artifact, no-features ineligible path, L0 with
markers that parse and markers that fail, L1 known-doc filtering, L1
`unsupported` when no known set is supplied, L2 detection of true
hallucinations, L2 vacuity for Non-RAG N=1, and recency detection with
mixed-offset citations.
