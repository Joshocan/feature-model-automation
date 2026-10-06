# Evaluation decisions and deviations — 0.1.0

Created 2026-09-26, after campaign generation and pilot analysis. No retrospective
preregistration claim is made. No generation files were changed by the specification freeze.

## Inherited rules

- Model/corpus/arm population: archived IFS 2027 matrices (694 planned runs).
- Cosine matching with all-mpnet-base-v2; tau=0.4 and sweep 0.3/0.4/0.5/0.6.
- Pilot independent-max primary and maximum-cardinality one-to-one sensitivity.
- Raw feature names, root-inclusive semantic nodes, raw similarities retained.
- MW U, Holm and Cliff's delta; no selection of parameters from outcome scores.
- Frozen trace suffix and metamodel switch behaviour remain generation facts.

## New evaluation clarifications (not earlier preregistration)

- Entire planned inventory is retained; metric eligibility is separate from outcome.
- Root depth=0 and only feature edges count toward depth.
- Run-level macro summaries are separate from pooled feature/citation summaries.
- Missing scores remain null with reasons, not zero or successful checkpoints.
- Stable occurrence IDs preserve duplicate nodes; alignment ties use reference order.
- Citation availability, attribution agreement and entailment are separate claims.
- Original provider attempts remain auditable; pilots never silently join main runs.
- H1a input capacity and output exhaustion are distinct. H4 is descriptive until
  a justified estimator is defined. Nonsignificance is not semantic equivalence.

## Decision register — resolved items and remaining freeze gates

### D01 — Metric eligibility and admissibility

The pilot gated semantics on its well-formedness/XSD and description/trace checks.
Using only that population for H3 can hide content in nonconforming ablated outputs.

**Proposed:** report two labelled semantic populations: strict-admissible finals
and all completed finals with unambiguous extractable features. Never repair XML,
invent missing nodes or score truncated checkpoints as finals. Publish both sample
counts and exclusions. Keep strict admissibility distinct from SAT and provenance
correctness. Decide which population is primary for each semantic comparison and
confirm the exact strict gate before implementing inferential comparisons.

Status: **resolved 2026-09-29 by author**. The primary semantic population is
all completed outputs with an extractable feature tree. Strict-admissible finals
are a separately labelled sensitivity population. The choice was made after
generation, and the paper must disclose that timing. Both populations retain
the planned-run outcome denominator; a missing semantic score is not zero.

### D02 — W1–W5 concrete-schema translation and logical tooling

The prompt's W5 distinguishes plain relations from formulas abstractly, but the
same prompt requires requires/excludes to be encoded as formula trees. Literal
"formula forbidden for requires" would reject the required XML representation.

**Proposed:** document an abstract-to-FeatureIDE rule mapping with canonical
requires/excludes/other-formula fixtures. Do not claim W5 checked until that
mapping is approved. Choose and pin the actual SAT backend and any FeatureIDE
parser; report unsupported checks honestly rather than equating XSD with FeatureIDE.

Status: **partial implementation, 2026-09-29**. The observable FeatureIDE XML
translation is now checked: requires is `<imp><var>A</var><var>B</var></imp>`,
excludes is `<disj><not><var>A</var></not><not><var>B</var></not></disj>`, and
every `<rule>` contains exactly one supported formula tree. Thus W5 forbids
extra or mixed formula material within a plain relation's rule. This
operationalisation overlaps W3/XSD; do not report it as independent evidence.
An actual FeatureIDE application parser remains unverified and is reported
separately as `unsupported`. If the paper promises parser acceptance, D02 is
not fully resolved until a pinned parser is integrated or that claim is removed.

### D03 — Matching encoder reproducibility

The generation encoder file leaves matching revision TBD. The pilot analysis
records revision e8c3b32edf5434bc2275fc9bab85f82640a19130, Transformers 4.57.6,
Torch 2.8.0 and a local mean-pooling encoder. This is evidence of the pilot
instrument, not proof that a complete identical snapshot is installed now.

**Proposed:** reuse that recorded revision and implementation; hash weights and
tokenizer, pin truncation/pooling settings, and verify a small known-score fixture.
Do not edit the generation hash manifest or substitute another encoder silently.

Status: **awaiting technical verification**, no download/call authorised by this phase.

### D04 — Statistical families and practical-effect claims

The inherited configuration names tests but not complete multiplicity families,
small-sample rules, confidence-interval methods, semantic equivalence margins or
a saturation estimator. There are empty and very small successful-output cells.

**Proposed:** list planned comparisons and families explicitly before scoring;
choose interval/minimum-n policies independently of favourable results. H3
equivalence and a numeric N* remain unavailable unless substantively justified
margins/estimators are approved. Otherwise report effects, uncertainty and
limitations, not "no effect". Same seed labels alone do not imply pairing.

Status: **concrete post-generation analysis specifications prepared 2026-09-30**
under the author's request to implement the follow-up analyses. See
[analysis-followup-plan.md](analysis-followup-plan.md) and the explicit family
JSON files under `config/analysis/families/`. Full inferential runs are left to
the author. Tau remains 0.4, no equivalence margin or numeric saturation
estimator is introduced, and the historical contract is not rewritten as
preregistration. These declarations follow descriptive inspection.

### D05 — Human validation and release scope

No expert agreement or corpus-support finding can be manufactured from cosine
scores. A targeted audit needs a sampling frame, sample size, rubric, blinding,
raters, disagreement procedure and rating-scale-appropriate agreement statistic.

**Proposed:** define a stratified audit across models, corpora and conformance
outcomes, or explicitly exclude the human-validation RQ from this release. Check
redistribution permissions separately before releasing corpus passages.

Status: **awaiting scope decision**. This does not block developing automated metrics,
but does block claiming a complete evaluation of the human-validation RQ.

## Resolving a decision

Append the selected rule, rationale, date, approver and affected metric IDs here.
Update the machine-readable contract and specification version together. Re-run
contract tests and input verification. Freeze only when all required decisions
are resolved (an explicitly approved exclusion is a resolution), with an archived
specification hash. Existing analysis drivers do not yet enforce these gates.
