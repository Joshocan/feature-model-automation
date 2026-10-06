# Parent-match and sibling agreement follow-up

This is an exploratory, post-generation audit, not a preregistered replacement
of parent-match. Existing scores, aggregates and hypothesis tests are unchanged.

The current parent-match implementation constructs a child correspondence in
`fame/evaluation/alignment.py:evaluate_alignment`: maximum cosine per generated
node, thresholded at tau, with the first reference occurrence breaking ties.
A child is evaluable only if it and its generated parent map, and the matched
reference child has a parent. Correctness means the mapped generated parent is
that reference parent. Thus this is **reference-parent agreement under semantic
node alignment**, not literal parent-name agreement and not general hierarchy
quality. Renamed grouping nodes can hurt it; zero alone does not diagnose why.

## Additional measure

Within matched non-root child occurrences (non-root on both sides), consider
unordered generated-node pairs with DISTINCT reference counterparts. Let:

- TP: siblings in both trees;
- FP: siblings only in the generated tree;
- FN: siblings only in the reference tree.

Sibling precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).
An empty denominator is null/not_applicable, not zero. F1 is zero if either
tree has eligible sibling pairs but none are shared. Report all denominators
and matching coverage. The suggested one-sided "sibling agreement" is recall;
a flat tree can score recall=1, so never report it without precision.

Sibling scoring does not compare parent labels, but child alignment remains
name/embedding-dependent. This is NOT fully label-independent tree comparison.
It evaluates grouping only, not parent meaning, complete topology, depth,
cross-tree constraints, AND/OR/ALT semantics, or unmatched content. Pair counts
are not independent statistical observations; comparisons must use runs as units.

Three policies are exported side by side:

1. independent_max: the existing thresholded child mapping. Same-reference
   collisions are excluded; duplicated matches can still reweight pairs.
2. one_to_one: maximum cardinality, then maximum total cosine; avoids many-to-one
   duplication but may change correspondences and the evaluated population.
3. exact_unique: exact identical names unique in BOTH trees, with no embedding
   threshold. This is an independent exact-name diagnostic, not the old
   embedding-filtered exact-parent metric. Duplicate names are excluded.

Do not assume the quoted 3/3 and 1/2 reproduce under every policy. Those
describe a small exact-name subset, whereas 0/28 and 0/208 parent scores use
larger semantic populations. Compare denominators before interpreting differences.
Sibling agreement alone cannot establish that a shared parent is domain-correct.

## Run

### Verified example checks (2026-10-01)

A read-only recomputation from the saved pairs reproduced existing parent counts
0/28, 0/208 and 0/19 for the three runs below. It also confirmed the quoted
exact-name sibling recalls, but NOT their generalisation to the semantic mapping:

| Run | Policy | TP / reference sibling pairs (recall) | TP / generated sibling pairs (precision) | Sibling F1 |
| --- | --- | --- | --- | --- |
| b143f8a9607b21eb | exact_unique | 3/3 | 3/3 | 1.000 |
| b143f8a9607b21eb | independent_max | 3/11 | 3/42 | 0.113 |
| 1f97a2cbc3c7fd86 | exact_unique | 1/2 | 1/3 | 0.400 |
| 1f97a2cbc3c7fd86 | independent_max | 26/889 | 26/711 | 0.033 |
| 023f9d17a2c38c75 | exact_unique | 0/0 (undefined) | 0/0 (undefined) | undefined |
| 023f9d17a2c38c75 | independent_max | 3/9 | 3/21 | 0.200 |

These are selected diagnostic cases, not full-campaign findings. Full exports
were not run during implementation. Seven unit tests passed.

All inputs are already in semantic-current-v3/pairs.csv, including occurrence
indices, names, parents and the full similarity matrix. No XML modifications,
embedding downloads or generation/API calls are needed. The script validates
saved topology and matrix completeness and streams one run at a time.

```bash
./.venv/bin/python scripts/evaluate_siblings.py \
  --pairs results/ifs-2027/analysis/semantic-current-v3/pairs.csv \
  --tau 0.4 \
  --output results/ifs-2027/analysis/sibling-current-v1
```

Optional small audit (still scans the saved CSV; no new embeddings):

```bash
./.venv/bin/python scripts/evaluate_siblings.py \
  --pairs results/ifs-2027/analysis/semantic-current-v3/pairs.csv \
  --run-id b143f8a9607b21eb 1f97a2cbc3c7fd86 023f9d17a2c38c75 \
  --output results/ifs-2027/analysis/sibling-examples-v1
```

Outputs:

- sibling_metrics.csv: run/policy-level counts, sibling P/R/F1, conditional
  parent-match and matched-node coverage. Join inventory metadata on run_id.
- node_alignment_audit.csv: child/parent names and indices on both sides,
  child and mapped-parent cosine, parent eligibility and correctness.
- summary.json: input/output hashes, code hashes, package versions, definitions.

Use fresh directories; the script refuses overwrites. A separate --tau 0.5 run
can provide sensitivity without changing primary tau=0.4. The exact_unique rows
are threshold-independent and must not be counted as additional repetitions.

Suggested next analysis: compare parent and sibling metrics on matched populations,
inspect positive as well as zero cases, then summarise at run level by campaign
cell. No new inferential family or change to the paper's claims is automatic.
