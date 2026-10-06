# Final automated analysis handoff

Prepared 2026-09-30 after generation and descriptive inspection. This is an
explicit post-generation analysis addendum, not preregistration. No generation
settings, reference mappings or stored campaign outputs are changed.

Execution update: matching commands below have now completed at the author's
request. `tau-matching-v1` contains 6,000 scores for 500 outputs across four
thresholds and three policies; `matching-comparison-v1` contains side-by-side
scores and headline rankings. Do not rerun into the same output directories.
Section 14 of [the results report](current-results-report-2026-09-30.md) records
the findings. Statistical family results still use the primary metric columns;
alternative-matching inferential robustness has not been established by this run.

## Decisions and limits

- Keep tau=0.4 primary; 0.3/0.5/0.6 remain sensitivity thresholds. Above-rho
  matches do not justify choosing a threshold that makes that pattern disappear.
- Keep independent-max primary and maximum-cardinality one-to-one sensitivity.
  The latter now maximizes total cosine weight among maximum-cardinality matches.
  This changes possible alignments, not its P/R/F1 counts. A third, separately
  labelled weight-only policy implements literal maximum-weight matching; it
  can choose fewer pairs. Never substitute its results silently for cardinality.
- A, B and C address distinct grounding, granularity and guidance questions.
  Their separation is justified by the questions, not a desire for significance.
  They provide within-family, not paper-wide, error control. If a paper-wide
  claim is required, apply an additional declared global adjustment.
- The specifications are prepared before these inferential runs but after
  descriptive outcomes were inspected. Disclose that timing.
- Four semantic endpoints are declared for every contrast: precision, closed
  reach recall, full-reference F1 and parent-match. Family C additionally tests
  structural conformance conditional on completion. The same two-sided MW-U
  test is used, including this binary conditional endpoint; interpret it as a
  distribution/rate comparison, not a test of means or semantic equivalence.
- Five eligible observations per arm are required for p-values and percentile
  mean intervals. Smaller cells retain counts and descriptive means, without
  p-values or intervals. This is a reporting floor, not a power guarantee.
- Bootstrap uses 10,000 resamples, seed 20260927, 95% percentile intervals.
  Intervals are pointwise; Holm adjusts p-values, not confidence intervals.
- Cliff's delta accompanies every tested contrast. Positive means arm A tends
  higher; A is RAG for grounding, larger N for granularity, guided for ablation.
- Every untestable declared comparison retains a Holm slot internally as p=1;
  no invented p-value is published for it. Missing output scores remain missing.
- Primary semantics use completed extractable outputs. Strict admissibility
  supplies a separately labelled sensitivity, not another chance to obtain
  significance. Equal seed labels do not imply paired tests.
- H3 equivalence and numerical H4 saturation remain outside these tests.

## Declared families

The JSON files in `config/analysis/families/` list every exact filter and metric.
They exclude retrieval sweeps and alternate-order runs from headline contrasts.

| Family | Configuration contrasts | Endpoints | Declared tests |
| --- | ---: | ---: | ---: |
| A grounding | 5: two corpora × two open models, plus Astra Repair | 4 | 20 |
| B adjacent granularity | 28: two open models × two conditions × (4 Repair + 3 Federation transitions) | 4 | 112 |
| C ablation | 10: two open models × two corpora × N={1,10}, plus Astra × two corpora at N=10 | 5 | 50 |

Astra Federation Non-RAG was not planned; do not invent that sixth grounding
contrast. Astra granularity is not part of family B. Strict sensitivity files
have the same tests, changing only the semantic population.

## Terminal commands

First generate all matching policies from saved pairs (no embeddings or paid calls):

```bash
./.venv/bin/python scripts/tau_rescore.py \
  --pairs results/ifs-2027/analysis/semantic-current-v3/pairs.csv \
  --tau 0.3 0.4 0.5 0.6 \
  --policy independent_max one_to_one one_to_one_max_weight \
  --output results/ifs-2027/analysis/tau-matching-v1

./.venv/bin/python scripts/compare_matching.py \
  --wide results/ifs-2027/analysis/aggregate-current-v4/wide.csv \
  --tau results/ifs-2027/analysis/tau-matching-v1/tau_sweep.csv \
  --output results/ifs-2027/analysis/matching-comparison-v1
```

`matching_comparison.csv` contains per-run side-by-side P/R/F1, differences,
feature-count ratios and precision ratios. `headline_rankings.csv` reports
mean-score ranks and rank changes within each corpus/grounding/N=10 comparison,
with successful and planned counts and median feature-count ratios. Empty
successful cells have no rank. These are descriptive ranks conditional on
completion, with tied mean scores receiving average ranks. Weight-only scores
are additional diagnostics; the rank-change flags compare primary with the
cardinality-first sensitivity. Inspect tau=0.4 first, then the entire sweep.

Run the declared primary and strict-sensitivity families sequentially:

```bash
for family in A_grounding B_granularity C_ablation; do
  ./.venv/bin/python scripts/analyse_family.py \
    --wide results/ifs-2027/analysis/aggregate-current-v4/wide.csv \
    --family "config/analysis/families/${family}.json" \
    --output "results/ifs-2027/analysis/family-${family}-v1" || break

  ./.venv/bin/python scripts/analyse_family.py \
    --wide results/ifs-2027/analysis/aggregate-current-v4/wide.csv \
    --family "config/analysis/families/${family}_strict.json" \
    --output "results/ifs-2027/analysis/family-${family}-strict-v1" || break
done
```

These tests use the primary matching metric columns in wide.csv. The matching
comparison command is descriptive; it does not automatically repeat hypothesis
tests under alternative matching policies or calculate alternative reach-based
recall. Do not claim inferential robustness to matching from rank tables alone.

Reproduce the bounded XML inspection (prints results without modifying outputs):

```bash
./.venv/bin/python scripts/inspect_ablation_xml.py
```

Use fresh output names if any requested output directory already exists.

## Strict admissibility and FeatureIDE scope

`strict_admissible` is a reproducible local gate, not a FeatureIDE application
acceptance certificate. It requires a completed output and successful XSD,
W1–W5, tree invariants, identifier syntax and XML-envelope checks, plus 100%
feature trace coverage and 100% L1 citation-ID validity. Missing checks yield an
unknown gate, not a pass. SAT, non-degeneracy and citation entailment are separate.

The semantic primary population does not apply this gate: excluding nonconforming
ablated outputs would otherwise discard semantic content. The strict sensitivity
asks whether conclusions change among outputs passing the documented gate.

For this handoff, the reporting scope is **local XML/schema/structural checks**.
FeatureIDE application import has not been tested and remains `unsupported`.
Report that limitation rather than relabelling XSD validation as FeatureIDE
acceptance. If application acceptance is a paper requirement, a pinned FeatureIDE
version must import the artifacts and produce separately recorded pass/fail
evidence. No application parser is installed or inferred by this change.
W5 uses the concrete formula-tree operationalisation; its overlap with W3/XSD
must not be presented as independent evidence. The frozen contract's unresolved
approval flags are not silently changed by this addendum.

## XML inspection: observed results

Selection rule: the first two completed Repair N=10 RAG DeepSeek Flash runs in
each condition, ordered by seed, repetition and run ID. This selected seeds 0
and 1 in both arms. Selection did not use a quality or degeneracy score. These
are completed XML outputs, not a restriction to strictly admissible outputs.

| Condition | Seed | Run ID | AND | OR | ALT | Leaf feature nodes | Non-root mandatory | Maximum depth |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Guided | 0 | 76eae3eb6c3292c4 | 83 | 0 | 7 | 219 | 0 | 3 |
| Guided | 1 | 9ad850a7529714a0 | 38 | 0 | 0 | 210 | 0 | 2 |
| Ablated | 0 | 2593f28c48d42cf8 | 10 | 0 | 0 | 508 | 9 | 2 |
| Ablated | 1 | 6f88f260168af9e2 | 17 | 0 | 1 | 547 | 0 | 4 |

Among group nodes, AND proportions are 92.2%, 100%, 100% and 94.4% respectively.
Root depth is zero. No selected description contains any of the eight exact
instruction phrases listed in the inspection script after case/whitespace
normalization. This limited lexical check does not exclude paraphrase or indirect
prompt influence. Generic words such as metamodel and formula appear in both
conditions and have legitimate domain meanings.

The current prompt hash matches all four recorded run configurations:
`472af198488c8818cbb34009cb0e9238f20e0278a9f4434c97c233402c2bc1ac`.
The optional block contains W1–W5, structural rules and the inserted XSD. The
always-present hierarchy/variability guidance also specifies when to use AND,
OR, ALT and mandatory features, including an AND fallback when choice semantics
are not established. Thus the ablation removes the optional metamodel block,
not every structural instruction. The XML skeleton does not itself show an
AND hierarchy. Do not attribute the observed difference exclusively to an AND
example in the optional block without further evidence.

Four outputs establish a descriptive contrast, not a causal explanation. The
guided files have more grouping nodes but are not uniformly devoid of ALT
groups; the ablated files have many more leaves and mixed mandatory/ALT usage.
Depth is also mixed. The proposed prompt-copying mechanism remains unconfirmed.

## Numeric dead-feature export and ablation commands

`structural__dead_features` contains JSON lists of names, not numbers. Use the
following to derive `structural__dead_feature_count` without rerunning SAT:

```bash
./.venv/bin/python scripts/export_dead_feature_counts.py \
  --wide results/ifs-2027/analysis/aggregate-current-v4/wide.csv \
  --output results/ifs-2027/analysis/dead-feature-counts-v1
```

This creates an augmented `wide.csv`, a compact `dead_feature_counts.csv` and
`summary.json`. Empty lists produce numeric zero; unavailable/ineligible metrics
remain blank with the original status and reason. The input is preserved. The
count is per output; summing counts across outputs counts occurrences, not unique
domain concepts. Use the derived CSV for external analysis or join on `run_id`.

The C-family declaration files already exist. The corresponding result directories
are only created after these commands finish successfully:

```bash
./.venv/bin/python scripts/analyse_family.py \
  --wide results/ifs-2027/analysis/aggregate-current-v4/wide.csv \
  --family config/analysis/families/C_ablation.json \
  --output results/ifs-2027/analysis/family-C_ablation-v1

./.venv/bin/python scripts/analyse_family.py \
  --wide results/ifs-2027/analysis/aggregate-current-v4/wide.csv \
  --family config/analysis/families/C_ablation_strict.json \
  --output results/ifs-2027/analysis/family-C_ablation-strict-v1
```

Strict filtering applies to semantic endpoints. Structural conformance retains
its metric-eligible population in both family runs; filtering that endpoint to
only admissible outputs would condition on passing conformance itself.

## Mapping-key request (status)

No identifiable list of 27 unresolved mapping keys was found in the searched
project data. No reference mappings or rho values were changed to improve a
result. Resolving those keys requires the actual source list and evidence for
each proposed mapping. The latest requested XML inspection is complete; it does
not resolve that separate, currently unspecified attribution task.
