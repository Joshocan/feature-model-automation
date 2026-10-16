# Cross-reference control (exploratory)

This post-hoc control scores each completed, extractable main-campaign output
against its own reference and the other corpus's reference. It is not a new
generation campaign, and does not change the registered primary threshold.

## Run

Restore the compatible data archive first (see [reproduce.md](reproduce.md)).
The original final XMLs, main inventory and both reference XMLs must be present.
The pinned `all-mpnet-base-v2` snapshot must already be cached locally; encoder
revision and package-version checks are the same as in `evaluate_semantic.py`.
No provider credentials, LLM calls or network model downloads are used.

```bash
./.venv-release/bin/python scripts/evaluate_cross_reference.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/cross-reference-v1
```

Use a new output directory every time. To check setup first, add `--limit 2` and
choose a separate directory such as `cross-reference-smoke-v1`. Limited runs are
explicitly marked partial and must not be reported as full Table 2 controls.

Default thresholds: `0.3 0.4 0.5 0.6 0.7 0.8`; override with `--tau` if needed.
Policies: `independent_max` and `one_to_one` (maximum cardinality, then maximum
total cosine similarity). The implementation calls the existing `rescore_run`;
it does not introduce a different matching rule. Features are node occurrences,
including duplicate names, using the original extractor and raw-name embeddings.

Both conditions are recomputed from XML. Generated embeddings are reused across
the two references within each run; reference embeddings are reused across runs.
Only one run's pair matrices are held at a time. Raw matrices are not exported;
the manifest records inputs and implementation hashes for regeneration.
Original evaluations are never overwritten. A failed evaluator aborts rather
than silently treating an error as zero. Noncompleted and unextractable outputs
are reported in `exclusions.csv`.

## Outputs

| File | Contents |
| --- | --- |
| `scores.csv` | One row per run, threshold and policy; own/cross P, R and F1, matched counts, reference sizes and own-minus-cross differences |
| `cell_summary.csv` | Unweighted means within corpus/model/arm/N/grounding/ordering/retrieval-depth/guidance cells; planned, completed and scored counts |
| `headline_rankings.csv` | Guided N=10, primary-order cells, RAG k=5 or Non-RAG; mean F1 and own/cross ranks within corpus and grounding |
| `exclusions.csv` | Ineligible, unextractable and smoke-limit omissions |
| `input_manifest.json` | Inventory, XML, contract and relevant implementation hashes |
| `summary.json` | Encoder identity, thresholds, matching objective, population, partial flag and output hashes |

For 500 eligible outputs the default produces 6,000 paired score rows. Full
reference denominators should be 100 for Repair and 130 for Federation; the
script records the counts actually extracted, not hard-coded denominators.

## Table 2 comparison

Use `headline_rankings.csv` at `tau=0.4`. Its own-reference mean F1, completed /
planned counts and median generated/own-reference size ratios should reconcile
with the supplied Table 2 before interpreting the cross scores. The population
is all completed, extractable outputs, not just strictly admissible outputs.
`guided_headline` and `astra_cross_corpus` are included; ablation, alternate
orderings and retrieval sweeps are excluded. Here `astra_cross_corpus` names the
existing Federation generation arm, not the opposite-reference scoring condition.

Rank using unrounded mean F1; exact ties receive average ranks. Planned cells
with no eligible output remain visible with null scores and no rank (including
any zero-completion Astra Non-RAG cell omitted from the paper table). All ranks
are descriptive; the two-completion DeepSeek Repair Non-RAG cell remains so.
Both conditions use exactly the same outputs within a cell. No new significance
tests or bootstrap intervals are produced by this script.

The other corpus is a related-domain comparison reference, not a guaranteed
unrelated null. Shared MDE concepts can match legitimately. Reference sizes also
differ: precision has the same generated denominator in both conditions but
still depends on the number of possible reference partners. Ranking persistence
or reversal alone does not prove a causal size effect or noise resistance.

The human pair-audit sampler and rating analysis are separate future work.
