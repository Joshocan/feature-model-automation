# IFS 2027 evaluation specification

Version: **0.1.0** · Date: **2026-09-26** · Status: **draft; not frozen**

Paper: *Conformance Is Not Correctness: Evaluating LLM-Constructed Feature Models*.

This is the initial evaluation specification, not a claim that the metrics
or reproduction pipeline are already implemented. It was written after generation
and pilot inspection. It is **not a preregistration**. The companion contract is
[`../config/evaluation/ifs-2027-v0.1.0.json`](../config/evaluation/ifs-2027-v0.1.0.json).
The metric dictionary and decision register below are normative companions.

## 1. Authority and change control

The archived campaign matrices and protocol describe what was generated. Existing
briefs describe intended analyses, not conclusions that must be obtained. This
specification documents inherited rules, new operational clarifications, and
unresolved decisions separately. Never edit generation prompts, matrices, reference
models, or the generation hash manifest to make evaluation pass.

Every evaluation release must record this specification version and hash, input
hashes, evaluator commit (and dirty-tree patch if applicable), dependency versions,
encoder weights/tokenizer hashes, and parameters. A semantic definition change
requires a new evaluation version and comparison note. Preserve old outputs.

## 2. Populations and experimental unit

Primary main-campaign population: `ifs-2027`, all **694 planned runs**, comprising
644 open-weight runs and 50 Astra runs in the archived matrices. The matrices,
not a directory glob or the presence of final XML, define membership. Do not pin
completion counts as inclusion rules: outcomes are measured by the run inventory.

The unit of replication is one planned run. Its steps, features, citations, and
provider attempts are nested observations, not independent repetitions. Keep
original run IDs and distinguish selected terminal attempt from recovery history.
Report initial-attempt incidents and recovery costs separately; do not erase them.

Group by corpus, provider/model, N, grounding, metamodel switch, ordering ID,
retrieval depth, and arm. Retain seed, repetition, generation settings and hashes.
Only combine cells through an explicit analysis definition; never pool all RAG
outputs across ablation, headline, order, or k-sweep arms.

Pilot population: the existing pilot correctness manifest, independently labelled.
Preserve all four isolated schema-depth pilot sources in that manifest (including
the GPT-OSS path whose name lacks `v2`). They are not additional main repetitions.
Other historical pilots are development evidence, not silently added comparison
observations. Reconcile any relocated archive using hashes and a path mapping.

## 3. Outcome and eligibility policy

Keep separate fields for completed generation, parseability, XSD validity,
W1–W5, structural conformance, output-contract compliance, provenance, SAT, and
degeneracy. Preserve the runner's original status alongside recomputed checks.

Completion requires all planned steps accepted by the generation protocol and
the corresponding final artifact. A final filename alone is insufficient.
Retain truncation, malformed XML, empty response, provider failure, input
infeasibility, interruption, missing artifact, and evaluator error separately.
`finish_reason=length` is output exhaustion, not proof of input infeasibility.

Use explicit metric states: `ok`, `not_applicable`, `ineligible`, `missing_artifact`,
`unsupported`, `evaluator_error`. A non-ok metric has a null value and a reason.
Zero is a measured score, not a substitute for unknown or failed evaluation.
Never substitute the last valid checkpoint for an incomplete final output.

Report completion and strict-admissible yield over **all planned runs**, plus
conditional admissibility among completed runs. Report non-degenerate yield over
planned runs and its conditional share among eligible outputs. Unknown checks
must be shown separately; absence of evidence is not a pass.

The author selected all completed finals with an extractable feature tree as the
primary semantic population on 2026-09-29, after generation. Strict-admissible
finals are a separately labelled sensitivity population. The current operational
strict gate requires XSD, W1–W5, tree/name/envelope checks, full feature trace
coverage and L1 referential integrity; its final approval remains part of D02.
SAT is a separate dimension. Preserve a pilot-compatible gate separately; do not
retroactively relabel historical pilot scores.

## 4. Feature extraction and semantic instrument

Inherited: `sentence-transformers/all-mpnet-base-v2`, cosine similarity, inclusive
threshold `similarity >= 0.4`, sensitivity thresholds 0.3/0.4/0.5/0.6, independent
maximum matching primary and one-to-one maximum-cardinality sensitivity.

The pilot recorded revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` using
`LocalTransformerEncoder`, Transformers 4.57.6 and Torch 2.8.0. Carry that identity
forward; do not select the newest local snapshot. Validate and hash actual weights,
tokenizer, pooling and truncation settings before freezing (D03). Matching and
retrieval encoders are distinct instruments.

Count named `and`, `or`, `alt`, and `feature` nodes beneath `struct`, including the
root and abstract nodes as the pilot did. Give every node a preorder index stable
within its artifact; duplicate names remain distinct occurrences. Use the raw XML
name as embedding input: no lowercasing, underscore replacement, synonym collapse,
or numeric-suffix removal. Record exact case-sensitive equality separately.
Descriptions do not contribute to name embeddings. Root-excluded metrics, if
reported, are explicitly supplementary, not silently substituted.

Store the full generated-by-reference cosine matrix with occurrence IDs and
parents, not only above-threshold matches. Keep unrounded values for scoring.
For generated nodes G and reference nodes T:

- P = number of G with at least one match in T / |G|.
- R_total = number of T with at least one match in G / |T|.
- F1_total = 2 P R_total / (P + R_total), or zero when P = R_total = 0
  and both feature populations are nonempty.
- Empty feature populations are ineligible/undefined, not perfect scores.
- R_S = matched reference nodes in subset S / |S|; an empty S yields null.

Reach definition correction (2026-09-30, post-generation): let S be reference
features whose attributed document IDs intersect the corpus manifest. Compute
`reach(T,C) = S union Ancestors_T(S)` using the named reference-feature parent
map, including the root when ancestral. The attested filter applies to S only;
organising ancestors are eligible. No unsupported sibling or descendant is added.
The semantic driver records `reach_definition=reference_ancestor_closure_v1`
and the exact closure membership in its summary. Earlier results used direct
attribution only and must not be pooled as the same reach metric. The frozen
inventory contract hash is retained for source identity; this documented
definition correction and implementation hashes must accompany new exports.

Report R_reach as the configured primary recall target, alongside R_total,
R_attested and R_organising. Label F1_total explicitly: do not silently pair P
against a different recall denominator. Attribution coverage rho is a property of
the annotated reference, not an established upper bound on recoverable semantics.

Independent matching can map several generated nodes to one reference node;
therefore report duplicate diagnostics and one-to-one sensitivity. For structural
and attribution alignment, select the highest-similarity reference node at or
above tau, ties by reference preorder index. Do not give exact matches an unrecorded
priority over the cosine rule.

## 5. Structure, conformance and provenance

Metric definitions and denominators are in [metric-dictionary.md](metric-dictionary.md).
Root depth is zero, child depth one. XML wrappers, descriptions and formula trees
never increase feature depth. Abstractness is the explicit attribute, not inferred
from having children.

For parent-match, a generated child must align above tau to a non-root reference
child; its generated parent must also align above tau. Correct means the parent's
selected reference ID equals the selected child's actual reference parent ID.
Publish `n_parent_correct`, `n_parent_evaluable`, exclusions and exact-name
sensitivity. Macro-average run rates; also label pooled counts as micro statistics.

Use the frozen `Trace: [id1, id2]` suffix grammar and corpus manifest IDs, never
filename-number guesses. Count missing descriptions/traces as omissions. Report
feature-level trace coverage and citation-pair-level integrity separately.
Track each feature–document pair's first appearance, not only feature birth.
New citations are checked against recorded current evidence; retained citations
may legitimately refer to earlier steps. Report current and cumulative exposure
separately, with uncertain lineage explicit. A document ID being listed does not
establish that its supporting passage was retrieved. Neither exposure nor reference
agreement proves textual entailment. Recency requires an exposure baseline and is
not intrinsically applicable to N=1.

## 6. Analyses and reporting boundaries

| Question | Eligible comparison and required reporting |
| --- | --- |
| H1a capacity | All planned runs and attempted steps; input/context margin separate from output cap, provider errors and completion. Unattempted steps are unknown. |
| H1b selection | Guided headline N=10, same model/corpus/ordering/k/settings, RAG versus Non-RAG; report semantic n and outcome rates. Empty cells are not testable. |
| H2 granularity | Guided baseline N=1, headline N=10 and guided curves at their registered N; no order/k/ablation pooling. Show reliability and semantic survivor selection together. |
| H3 guidance | Guided versus ablated RAG at matching model/corpus/N/settings; completion, conformance, primary extractable-tree semantics, strict sensitivity, variability and cost separately. Equivalence policy remains D04. |
| H4 saturation | Descriptive N-curves and within-run trajectories, with accumulated evidence fraction and attrition. Do not infer semantic saturation from failures or a universal scaling law from two corpora. |
| Criterion divergence | Separate structural, logical and semantic dimensions; expose valid-but-reference-inaccurate cases. Do not require rankings to differ. |
| Provenance/variability | Both reference-free and reference-relative measures with their own eligible denominators; no implication that many citations or nodes means correctness. |
| Robustness | Separate order N=10/54 and k sweeps; tau and matching sensitivity; model/run variation, not node-level pseudo-replication. |
| Human validation | Manual evidence-support and matcher audit only after a documented sample/rubric/rater plan (D05). No automated substitute for expert ratings. |

Inherited statistical tools: mean with 95% interval, Mann–Whitney U, Holm correction,
Cliff's delta. Also report raw run points, median, and effective n. These tools do
not constitute a two-factor test and must not be used to claim an interaction.
Pairing requires a justified design, not merely the same numerical seed label.
No interval at n=1 may be presented as zero uncertainty. Do not interpret a
non-significant comparison as equivalence or as proof that retrieval buys only
capacity. Final families, minimum-n policy, interval construction and any
equivalence/saturation margins remain D04 decisions. No inferential claim is frozen
by this draft. Never tune tau, k, N, weights or inclusion from resulting rankings.

## 7. Outputs and freeze gate

Each metric table needs evaluation version, run and artifact IDs, population label,
value, numerator/denominator where applicable, status and exclusion reason.
Each comparison needs the full cell keys, included run IDs, n, effect size,
uncertainty, raw/adjusted p where applicable and multiplicity family.
All figures derive from those tables; missing points remain missing. Pilot and
main tables are separate. Evaluator failures must be visible in the run summary
and cause a non-successful evaluation exit, never plausible-looking blank results.

Before changing status to frozen:

1. Resolve D01–D05, recording decisions and dates without backdating.
2. Verify archived inputs and encoder identity; archive this specification and hash.
3. Validate the metric dictionary against fixtures and reconcile all planned runs.
4. Pin execution dependencies and evaluator implementations, including SAT and any
   actual FeatureIDE parser. An unavailable parser stays explicitly unmeasured.
5. Implement a driver that reads this contract and refuses unresolved required
   decisions. The current campaign analysis scripts do **not** enforce it yet.

Specification drafting does not run campaigns, score outputs, delete artifacts, or
rewrite the generation freeze. Subsequent implementation follows this specification
only after the affected decisions are resolved.
