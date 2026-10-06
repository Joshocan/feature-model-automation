# Current campaign results: Conformance Is Not Correctness

Date: 2026-09-30. Status: descriptive analysis for manuscript preparation; not a final inferential report.

**Current verified results:** This report uses `semantic-current-v3` and
`aggregate-current-v4`, with `reach_definition=reference_ancestor_closure_v1`.
All 694 run identities are unchanged, and all 500 completed outputs have reach
recall, with zero semantic evaluator errors. Reach includes directly supported
reference features and all their ancestors: Repair has 84/100
(58 direct + 26 ancestors; rho=0.84), and Federation has 59/130
(36 direct + 23 ancestors; rho=0.4538461538).

Relative to the previous direct-only results, only reach size, rho and reach
recall changed. Precision, full-reference recall, F1, parent-match, conformance,
provenance and criterion correlations are unchanged. Previous versions remain
historical artifacts and must not be used for the current reach definition.

Recall above rho is not itself evidence of unsupported generation. The correct
decomposition is `R_total = rho * R_reach + (1-rho) * R_outside_reach`, where
the complementary target is `F_t minus reach`, not the organising partition.
Matches outside reach require evidence and matching audits. A score below rho
also does not prove that every matched feature lies inside reach.

## 1. Scope and source data

This report describes the saved IFS 2027 campaign evaluations after correcting provenance manifest loading and correlation handling. It does not report new generation, newly executed significance tests, or expert ratings. Numbers are rounded from the saved run-level tables. The unit of observation for semantic summaries is a completed, extractable final output, not an individual feature or an intermediate step.

Authoritative inputs for this report:

- [Inventory](../results/ifs-2027/analysis/inventory-current-v1/summary.json): the 694 planned runs and selected outcomes.
- [Structural evaluation](../results/ifs-2027/analysis/structure-current-v1/summary.json): conformance, satisfiability and variability.
- [Semantic evaluation](../results/ifs-2027/analysis/semantic-current-v3/summary.json): corrected reach recall, reference matching and hierarchy metrics.
- [Corrected provenance evaluation](../results/ifs-2027/analysis/provenance-current-v2/summary.json): citation identifiers and context checks.
- [Corrected aggregation](../results/ifs-2027/analysis/aggregate-current-v4/summary.json) and [run-level wide table](../results/ifs-2027/analysis/aggregate-current-v4/wide.csv).
- [Threshold sensitivity](../results/ifs-2027/analysis/tau-current-v1/summary.json): four thresholds using independent-max matching.

The regenerated semantic pairs file has SHA-256
`d892c5d79a3c8f722b3ce5dbaf39ca1c620ef4420bc54c56df3cb4407997823e`,
identical to the input hash recorded by `tau-current-v1`. The existing threshold
sweep therefore remains applicable after the reach correction.

The primary matching threshold is cosine similarity ≥0.4, using the recorded all-mpnet-base-v2 encoder. Semantic precision, recall and F1 are automated reference-based proxies; they are not direct measurements of expert-assessed correctness. Full-reference recall and F1 are reported below, alongside the restored primary reach-based recall.

Pilots, recovery attempts and the proposed expert-study extension must retain their separate population identities. Recovered attempts are not additional independent repetitions.

## 2. Executive interpretation

The observed results support investigating several distinct dimensions of quality. Completion does not guarantee conformance; conformance does not guarantee satisfiability; valid citation identifiers do not establish correct attribution; and feature-name similarity does not guarantee correct hierarchy.

Retrieval generally improves completion in the headline comparisons shown here, but its semantic advantage varies by corpus and model. Increasing the number of batches can increase semantic quality among surviving outputs while reducing the probability of obtaining a final output. Metamodel guidance does not produce a uniform improvement across all metrics. These findings support a multidimensional evaluation rather than a single best-model ranking.

These interpretations are descriptive. Confidence intervals, declared comparison families and robustness analyses remain necessary before making formal hypothesis claims.

## 3. Campaign disposition and evaluation denominators

| Recorded outcome | Runs | Percentage of 694 planned |
| --- | ---: | ---: |
| Completed | 500 | 72.0% |
| Truncated output | 93 | 13.4% |
| Malformed XML | 100 | 14.4% |
| Empty response | 1 | 0.1% |
| Total | 694 | 100% |

Percentages may not sum exactly because of rounding. All planned runs have a recorded selected outcome in this inventory. That does not imply all evaluation or publication checks are complete.

| Evaluation stage | Result | Denominator and interpretation |
| --- | ---: | --- |
| Semantic scoring | 500 outputs | All completed outputs were scored |
| XSD validity | 381 pass; 119 fail | 500 completed outputs; 76.2% pass |
| Satisfiability | 369 SAT; 12 UNSAT | 381 evaluated outputs; 96.9% SAT |
| Degeneracy | 106 flagged; 275 unflagged | 381 evaluated outputs; 27.8% flagged |

XSD-valid outputs represent 54.9% of all planned runs. This is different from the conditional 76.2% validity rate among completed outputs. SAT and degeneracy were not measured on all 694 runs; their missing values must not be treated as successful checks or negative findings. Non-degeneracy and SAT are separate properties, so their pass counts must not be multiplied to infer a joint success rate.

The primary semantic population follows the author's recorded D01 decision: completed outputs with extractable feature trees. Strict-admissible outputs form a separate sensitivity population. Failed runs retain missing semantic scores, and the last valid intermediate checkpoint is not substituted for a failed final output.

## 4. Headline semantic comparisons at N=10

Values are means of run-level scores among completed outputs. Mean F1 is the mean of individual run F1 values, not the harmonic mean of the displayed mean precision and recall.

| Model | Corpus | Condition | Completed/planned | Precision | Full-reference recall | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek Flash | Repair | RAG | 16/20 | 0.731 | 0.693 | 0.710 |
| DeepSeek Flash | Repair | Non-RAG | 2/20 | 0.679 | 0.730 | 0.703 |
| DeepSeek Flash | Federation | RAG | 12/20 | 0.681 | 0.573 | 0.622 |
| DeepSeek Flash | Federation | Non-RAG | 8/20 | 0.646 | 0.638 | 0.641 |
| GLM Flash | Repair | RAG | 18/20 | 0.722 | 0.511 | 0.597 |
| GLM Flash | Repair | Non-RAG | 14/20 | 0.423 | 0.572 | 0.483 |
| GLM Flash | Federation | RAG | 17/20 | 0.641 | 0.429 | 0.514 |
| GLM Flash | Federation | Non-RAG | 14/20 | 0.585 | 0.586 | 0.585 |
| Astra | Repair | RAG | 7/10 | 0.620 | 0.804 | 0.700 |
| Astra | Repair | Non-RAG | 0/10 | — | — | — |
| Astra | Federation | RAG, cross-corpus arm | 10/10 | 0.607 | 0.699 | 0.650 |

Astra's Federation result comes from the separately labelled cross-corpus arm. There is no corresponding Astra Federation Non-RAG comparison in this table.

Interpretation:

- DeepSeek Repair has similar conditional F1 across conditions but very different completion rates. The two successful Non-RAG outputs are insufficient for a reliable generalization about that condition's semantic performance.
- GLM's RAG precision is higher on both corpora, while recall is lower. The resulting F1 advantage changes direction between corpora. This is a reason to report precision and recall separately.
- Astra's Repair RAG recall exceeds that of the other RAG models in this table, while its precision is lower. Its failed Non-RAG runs provide no final semantic scores.
- Higher conditional F1 does not establish better overall deployment reliability. Present semantic scores alongside outcome rates.

## 5. Corrected reach recall and criterion correlations

### Corrected reach-based recall

| Corpus | Completed outputs | Reach/reference size | Mean reach recall | Mean organising recall | Mean full-reference recall | Outputs with full-reference recall above ρ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repair | 267 | 84/100 | 0.5341 | 0.4626 | 0.5192 | 0/267 (0%) |
| Federation | 233 | 59/130 | 0.5005 | 0.3452 | 0.3848 | 84/233 (36.1%) |

These are descriptive run-level means pooled over different configurations,
not controlled cross-corpus comparisons. Reach includes some organising ancestors,
so reach recall no longer equals attested recall. Organising recall remains a
separate metric and is not the complement of closed-reach recall.

The earlier direct-only definition produced above-rho frequencies of 98/267 for
Repair and 124/233 for Federation. With ancestor closure, the Repair excess
disappears and Federation falls to 84/233. These frequencies describe automated
matches, not proven hallucination or unsupported generation.

Primary reach recall at headline N=10, among completed outputs:

| Model | Corpus | RAG reach recall (n) | Non-RAG reach recall (n) |
| --- | --- | ---: | ---: |
| DeepSeek Flash | Repair | 0.701 (16) | 0.744 (2) |
| DeepSeek Flash | Federation | 0.664 (12) | 0.765 (8) |
| GLM Flash | Repair | 0.530 (18) | 0.591 (14) |
| GLM Flash | Federation | 0.551 (17) | 0.736 (14) |
| Astra | Repair | 0.801 (7) | — (0) |
| Astra | Federation, cross-corpus | 0.807 (10) | Not planned |

In these observed open-weight headline cells, conditional reach recall is lower
under RAG, although precision is higher in the headline table. Thus H1b needs
metric-specific interpretation: retrieval may favor precision over coverage.
The differences are not significance findings and are affected by unequal
completion rates, especially the two-output DeepSeek Repair Non-RAG cell.

### Correlations after regeneration

Both correlation coefficients now use the same finite observed pairs of configuration-cell scores. Missing criteria are excluded from the pair rather than ranked as low performance.

| Criteria | Eligible cells | Excluded cells | Spearman ρ | Kendall τ-b |
| --- | ---: | ---: | ---: | ---: |
| Structural conformance / logical satisfiability | 48 | 17 | 0.013 | 0.012 |
| Structural conformance / semantic F1 | 57 | 8 | −0.263 | −0.196 |
| Logical satisfiability / semantic F1 | 48 | 17 | 0.119 | 0.100 |

The corrected coefficients replace the values in `aggregate-current-v1`. They show that the aggregate criteria do not induce identical assessments. They do not establish a causal trade-off or statistical significance. The cells mix corpora, models and arms, have unequal sample sizes, and use different metric eligibility populations. Report within-stratum comparisons and uncertainty before making a general statement about their relationship.

SAT is also often near its ceiling; a weak association with another metric may partly reflect limited variation. Concrete examples of conforming but semantically weak models and nonconforming but semantically informative outputs will strengthen interpretation more than a pooled coefficient alone.

## 6. Citation validity and attribution

The corrected manifest reader loads 54 Repair and 23 Federation document IDs. Provenance evaluation reports zero evaluator errors.

| Measure | Current finding |
| --- | --- |
| L1 eligible completed outputs | 495 |
| L1 not applicable among completed outputs | 5 |
| Mean per-run L1 referential integrity | Approximately 99.92% |
| Outputs with L1 below 100% | 5 |
| Mean reference-attribution agreement | Approximately 40.45%, across 495 applicable outputs |
| Citation rows in detailed output | 123,521 |
| Rows in the context-hallucination output | 43 |

The 99.92% result is a mean of run-level proportions, not a pooled percentage of citation rows. The 43 context-hallucination rows are not a measure of all unsupported semantic claims, and should not be divided by the citation-row total without checking the applicable L2 denominator.

L1 establishes that an identifier belongs to the corpus. L2 asks whether a cited document was available in context under the implemented first-seen policy. Neither verifies that the document entails the feature. Reference-attribution agreement depends on feature alignment and reference annotations; a disagreement is not automatically evidence that the model fabricated support. Expert review should distinguish invalid identifiers, unavailable evidence, plausible alternative attribution and genuinely unsupported claims.

## 7. Granularity and reliability

DeepSeek Flash, Repair RAG illustrates the main trade-off:

| N | Completed/planned | Mean full-reference F1 |
| ---: | ---: | ---: |
| 1 | 20/20 | 0.472 |
| 5 | 5/5 | 0.616 |
| 10 | 16/20 | 0.710 |
| 20 | 3/5 | 0.724 |
| 54 | 0/5 | — |

Quality among surviving outputs increases over the observed successful cells, while completion declines. This does not locate a saturation point at N=20: that cell contains only three successful outputs, and N=54 contains none.

GLM Repair RAG is not monotonic: mean F1 is 0.513 at N=1, 0.435 at N=5, 0.597 at N=10, 0.660 at N=20 and 0.635 at N=54. The corresponding completion counts are 17/20, 3/5, 18/20, 3/5 and 2/5. Small surviving populations limit the strength of curve-based conclusions.

Granularity figures should place a quality curve and a completion curve together. Display eligible counts and confidence intervals where estimable, and leave missing scores as gaps. Do not connect a missing cell to zero quality or extrapolate through it.

## 8. Metamodel ablation

Repair N=10 RAG provides the following comparison:

| Model | Guided completed/planned | Ablated completed/planned | Guided F1 | Ablated F1 | Guided conformance among completed | Ablated conformance among completed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek Flash | 16/20 | 19/20 | 0.710 | 0.706 | 13/16 | 14/19 |
| GLM Flash | 18/20 | 12/20 | 0.597 | 0.513 | 16/18 | 7/12 |
| Astra | 7/10 | 9/10 | 0.700 | 0.705 | 7/7 | 9/9 |

The differences are model-dependent. DeepSeek and Astra show small descriptive F1 differences, whereas GLM shows a larger reduction under ablation. Astra does not show the predicted conformance drop in this Repair comparison. A universal claim that removing guidance sharply lowers conformance is therefore not supported by these descriptive results alone.

Nonsignificant semantic differences, if later observed, would not establish equivalence. That claim requires a substantively justified equivalence margin and suitable uncertainty analysis. Outcome-dependent missingness also complicates comparisons between surviving outputs.

## 9. Variability and hierarchy

Among the 381 structurally evaluated outputs, 106 are flagged degenerate. By model, the counts are:

| Model | Degenerate/evaluated | Conditional share |
| --- | ---: | ---: |
| DeepSeek Flash | 99/166 | 59.6% |
| GLM Flash | 7/182 | 3.8% |
| Astra | 0/33 | 0.0% |

These are pooled descriptive summaries over different campaign mixes, particularly for Astra; they are not balanced model rankings. The operational flag means no OR group, no ALT group and no non-root mandatory feature. It is not a blanket measure of semantic uselessness, and appropriate variability also depends on the evidence.

One notable case warrants XML-level inspection: DeepSeek Repair N=10 RAG has 12/13 guided structurally evaluable outputs flagged degenerate, compared with 1/14 ablated outputs. Verify representative examples and the implementation before advancing a claim about why guidance changes variability.

Mean run-level parent-match across the 500 scored outputs is approximately 1.46%. The mean number of evaluable generated nodes is approximately 81 per output, but the denominator varies substantially. This suggests the low aggregate value is not explained entirely by tiny samples. It does not remove uncertainty from embedding alignment, duplicate mappings or sparse individual cells.

Report parent-match with evaluable-node counts, unique reference coverage and exact-name sensitivity. Correct hierarchy is a separate question from depth: deeper models need not reproduce the reference's parent relationships.

## 10. Implications for the hypotheses

| Hypothesis | Current interpretation | Required next evidence |
| --- | --- | --- |
| H1a: capacity | Completion differences are observed, but input infeasibility is not established by output truncation | Assembled-input budgets, model windows and explicitly classified capacity failures |
| H1b: retrieval selection | Semantic advantage is mixed across models and corpora | Matched-configuration comparisons, precision/recall decomposition, uncertainty and Holm-adjusted tests |
| H2: granularity | Some curves improve; others are nonmonotonic, with increasing failure | Model/corpus-specific contrasts and joint reporting of quality and completion |
| H3: metamodel guidance | Effects vary by model and metric; sharp conformance loss is not universal | Guided/ablated contrasts at matching configurations and strict-population sensitivity |
| H4: saturation | Some conditional gains diminish, but missing cells and small samples prevent a precise N* claim | An explicit estimator or a clearly descriptive treatment without numerical saturation claims |

Hypotheses should be allowed to receive mixed or contrary evidence. Do not select only favorable configurations, thresholds or populations after inspecting the outcomes.

## 11. Remaining analysis and figures

Recommended sequence:

1. Audit cases behind low parent-match, attribution disagreement and unusual degeneracy. Confirm denominators and inspect raw XML and evidence.
2. Declare statistical comparison families, eligible populations, minimum-sample handling and interval methods. Record these as post-generation decisions. Seed labels alone do not justify paired tests.
3. Generate run-level precision, recall, F1 and hierarchy summaries with dispersion and uncertainty, alongside planned-run completion and conformance rates.
4. Compare the primary completed-extractable population with the strict-admissibility sensitivity population.
5. One-to-one matching and descriptive headline ranking sensitivity are now complete (see section 14). Hypothesis-test robustness under alternative matching still requires separate analysis; existing family tests use independent-max columns.
6. Analyse order sensitivity and retrieval depth separately, matching configurations and retaining the planned arm identities.
7. Produce the paper figures and a table mapping each published number to its source and command.

Suggested figures are: outcome rates by configuration; quality and completion versus N; guided/ablated effect estimates with intervals; criterion-disagreement plots; and citation integrity alongside attribution measures. Threshold and retrieval-depth sensitivity plots can appear in the appendix. Reference-based group-kind/mandatory agreement, input-capacity plots and intermediate growth curves must be checked or derived separately rather than assumed complete from the current aggregate table.

Current summaries remain `publication_ready: false`. Open items include the strict sensitivity gate/FeatureIDE-parser scope, statistical-family definitions and human-validation requirements. Some older semantic summaries retain historical D01 labels; these do not reverse the author's resolved population decision. Preserve historical artifacts and record corrections in newly generated outputs.

### Reporting readiness

Implementation update: [analysis-followup-plan.md](analysis-followup-plan.md)
contains the prepared matching/ranking commands, exact A/B/C comparison families,
strict-gate scope and the completed four-output XML inspection. The matching
sweep and ranking comparison have now been run at the author's request;
the assistant has not rerun statistical families under alternative matching. The four-output
audit does not establish metamodel-copying as a causal explanation. The claimed
27 unresolved mapping keys remain unverified pending an identifiable source list.

The manifest, correlation-population and reach-definition bugs identified in this
review have been corrected. This does not establish that every evaluation
component has been fully audited. Descriptive manuscript drafting can proceed
using the current outputs while the following checks are completed:

| Priority | Remaining work | Reporting consequence |
| --- | --- | --- |
| 1 | Audit representative matches, especially Federation matches outside closed reach | Do not label above-rho frequencies as unsupported-generation rates |
| 2 | Interpret the completed matching/ranking sensitivity; test inferential robustness separately | Do not transfer primary-policy p-values to one-to-one scores |
| 3 | Declare and execute statistical comparison families, effect sizes and intervals | Defer statistical superiority and hypothesis-test conclusions |
| 4 | Finalize strict-admissibility sensitivity and FeatureIDE-parser scope | Distinguish implemented checks from unsupported acceptance claims |
| 5 | Complete expert evaluation | Defer claims of expert-assessed correctness and evidence support |

Completion/failure counts, measured conformance and SAT rates, degeneracy with
explicit denominators, descriptive semantic scores, closed-reach calibration
and descriptive correlations can already be reported. Do not claim semantic
equivalence, a precise saturation point or statistically established superiority
from the current descriptive tables. The automated checks above use existing
outputs; they do not require new generation campaigns. Expert evaluation remains
a separate study with its own sampling and evidence requirements.

## 12. CSV exports for further analysis

### Primary export

Export [aggregate-current-v4/wide.csv](../results/ifs-2027/analysis/aggregate-current-v4/wide.csv) first. It contains one row per planned run (694 rows), configuration fields, metric values, eligibility/status fields and failure outcomes. This is the main file for further analysis in R, Python or statistical software.

Join [inventory-current-v1/runs.csv](../results/ifs-2027/analysis/inventory-current-v1/runs.csv) by `run_id` when additional inventory/configuration fields are needed. Verify that each input has unique run IDs and that the join preserves 694 rows.

### Supplementary exports

| File | Intended use |
| --- | --- |
| `aggregate-current-v4/cell_summary.csv` | Configuration means, dispersion and metric-specific counts; useful for tables, not a replacement for run-level inference |
| `aggregate-current-v4/variability.csv` | Group/mandatory and degeneracy summaries |
| `aggregate-current-v4/correlations.csv` | Corrected cell-level correlations, eligible counts, exclusions and status |
| `tau-current-v1/tau_sweep.csv` | 2,000 score rows: 500 outputs × four thresholds; join configuration metadata by run ID |
| `semantic-current-v3/metrics.csv` | Long-format semantic results, retaining metric-specific status and detail |
| `structure-current-v1/metrics.csv` | Long-format structural, logical and variability results |
| `provenance-current-v2/metrics.csv` | Corrected long-format provenance measures |
| `provenance-current-v2/citations.csv` | 123,521 citation records for attribution and first-seen inspection |
| `provenance-current-v2/hallucinated.csv` | Detailed records flagged by the context-availability check |
| `semantic-current-v3/pairs.csv` | 9,519,810 generated/reference similarity pairs for matching audits and re-thresholding |
| `tau-matching-v1/tau_sweep.csv` | 6,000 scores: 500 completed outputs × four thresholds × three matching policies |
| `matching-comparison-v1/matching_comparison.csv` | 2,000 run/threshold rows with side-by-side P/R/F1, differences and ratios |
| `matching-comparison-v1/headline_rankings.csv` | 40 model/corpus/grounding/threshold rows with conditional mean-score ranks and rank changes |

The pairs file exceeds a standard Excel worksheet's row limit. Analyse it using Python, R or a database; do not rely on a spreadsheet import that may truncate it. Similarity pairs, citation rows and repeated thresholds are not independent experimental repetitions.

Keep the relevant `summary.json` files and the evaluation contract with exported CSVs so hashes, source versions, eligibility rules and matching settings remain traceable. CSV export alone is not a complete reproducibility package.

For all analyses, filter metric values using their status columns, preserve missing values, report both planned and evaluable counts, and avoid counting the same run repeatedly after a one-to-many join. Do not average all model outputs together to produce a model league table: Astra has a different arm composition from the open-weight models.

## 13. Manuscript-ready descriptive paragraph

> Across 694 planned runs, 500 produced completed final outputs (72.0%). Of these, 381 passed XSD validation (76.2% of completed outputs; 54.9% of planned runs). Among the 381 outputs evaluated for satisfiability, 369 were satisfiable, while 106 were flagged by the operational degeneracy criterion. Corrected configuration-level correlations showed weak agreement between structural conformance, logical satisfiability and reference-based semantic F1. Citation referential integrity was high, averaging approximately 99.92% across 495 applicable outputs, whereas mean reference-attribution agreement was approximately 40.45%. These measures assess different properties and use different eligibility rules. Headline comparisons further showed that retrieval and granularity affected completion and conditional semantic quality differently across models and corpora. The findings motivate reporting reliability, conformance, variability, semantic coverage, hierarchy and provenance separately rather than treating successful parsing or execution as evidence of correctness.

> Under the ancestor-closed reach definition, 84 of 100 Repair reference features and 59 of 130 Federation reference features were reachable. Mean reach recall among completed outputs was 53.41% and 50.05%, respectively, pooled across configurations. Full-reference recall exceeded the corresponding annotated reach ratio in 0/267 Repair and 84/233 Federation outputs. These frequencies describe automated reference matching and do not establish unsupported generation.

This paragraph is descriptive. Add final confidence intervals, validated hypothesis results and expert findings only after those analyses are complete.

## 14. Executed matching sensitivity

At the author's request, the saved 9,519,810 similarity pairs were rescored for
all 500 completed outputs at tau={0.3,0.4,0.5,0.6}. Three policies were evaluated:
independent-max; maximum-cardinality one-to-one with maximum-weight tie-breaking;
and separately labelled weight-only bipartite matching. No model or embedding
calls were made. All 2,000 independent-max P/R/F1 triplets reproduce the existing
threshold sweep to numerical comparison tolerance 1e-12. The new sweep contains
6,000 unique run/threshold/policy records. All one-to-one match counts are bounded
by the smaller generated/reference population.

The following table uses cardinality-first one-to-one at tau=0.4. Precision and
F1 are means across completed outputs; ranks are within the same corpus and RAG
condition, among the three models with observations.

| Corpus | Model | Completed/planned | Independent precision | One-to-one precision | Independent F1 | One-to-one F1 | F1 rank change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Repair | DeepSeek Flash | 16/20 | 0.731 | 0.253 | 0.710 | 0.366 | 1 → 2 |
| Repair | GLM Flash | 18/20 | 0.722 | 0.409 | 0.597 | 0.439 | 3 → 1 |
| Repair | Astra | 7/10 | 0.620 | 0.135 | 0.700 | 0.231 | 2 → 3 |
| Federation | DeepSeek Flash | 12/20 | 0.681 | 0.254 | 0.622 | 0.341 | 2 → 2 |
| Federation | GLM Flash | 17/20 | 0.641 | 0.362 | 0.514 | 0.366 | 3 → 1 |
| Federation | Astra | 10/10 | 0.607 | 0.160 | 0.649 | 0.258 | 1 → 3 |

At tau=0.4, Non-RAG F1 rankings also reverse: GLM exceeds DeepSeek on both
corpora under one-to-one matching. The Repair DeepSeek comparison has only two
completed outputs and remains descriptive. Astra Repair Non-RAG has no final
output and receives no rank. Across the ten scored headline model/corpus/
grounding entries, F1 ranks change in 8 at tau=0.3, 9 at 0.4, 9 at 0.5 and 7
at 0.6. These counts describe rank positions, not independent statistical tests.

Astra's current Repair headline median generated feature count is 605 against
100 reference features. Its median per-run independent/one-to-one precision
ratio is 4.625. Federation Astra has a median 548.5 generated features against
130 reference features and a median precision ratio of approximately 3.788.
Ratios are medians of run ratios, not ratios of the displayed means. These
numbers describe the current campaign, not earlier pilot medians.

The evidence establishes substantial scoring-policy sensitivity and ranking
reversals. It does not establish that every feature losing credit is incorrect:
one-to-one matching also penalizes finer granularity relative to the reference.
Many-to-one credit is therefore an important measurement effect to disclose, not
by itself a human-validated error rate. Rankings remain conditional on completion
and use unequal survivor populations. Keep independent-max primary as declared,
report this sensitivity alongside it, and do not reuse primary-policy p-values
as tests of the one-to-one results.

Artifacts: [side-by-side scores](../results/ifs-2027/analysis/matching-comparison-v1/matching_comparison.csv),
[headline rankings](../results/ifs-2027/analysis/matching-comparison-v1/headline_rankings.csv),
and [full sweep summary](../results/ifs-2027/analysis/tau-matching-v1/summary.json).
