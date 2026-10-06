# Data dictionary

## File and row interpretation

`protocol/tabular-schemas.json` enumerates actual archived CSV headers, delimiter
and file hash. It is a schema catalogue, not an inference of undocumented units.
Read it with the metric definitions in `protocol/documentation/metric-dictionary.md`
and evaluator-specific documentation in that directory. Historical decision labels
must be interpreted using `protocol/deviations.md` and `protocol/analysis-lineage.json`.

| File family | Unit / interpretation | Key |
| --- | --- | --- |
| corpus manifests | One document, title, filename, bibliography and source link; semicolon-delimited | corpus + doc_id |
| attribution | Reference feature/document attribution records; not expert entailment judgments | corpus + feature identifier + doc_id |
| feature partitions | Declared reference-feature partitions | corpus + reference feature identifier |
| frozen matrices | One planned configuration/run | campaign_id + corpus + run_id |
| run_meta.json | One current recorded run outcome and ordered step records | campaign_id + corpus + run_id |
| context_log.jsonl | Per-step context/usage records as emitted; use stored step index | run_id + step_index |
| fm_iter/step_XX.xml | Accepted intermediate checkpoint; XX is zero-based | run_id + step |
| step_XX.raw.txt | Saved unaccepted/raw response, where available; not a final model | run_id + step |
| fm_gen.xml | Final artifact of a completed run | run_id |
| inventory runs.csv/json | All planned main-campaign rows, including failures | run_id |
| evaluator metrics.csv/json | Per-run evaluator outputs; inspect header/schema for exact shape | run_id |
| semantic pairs.csv | Generated/reference node similarity candidates within a run | run_id + occurrence indices |
| wide.csv | Joined per-run metrics with evaluator prefixes and status/reason columns | run_id |
| cell_summary.csv | Configuration-level summaries, not independent observations | declared cell grouping columns |
| tau_sweep.csv | Per-run, threshold and matching-policy results | run_id + tau + policy |
| family outputs | Declared contrast/test results and eligibility; not raw independent samples | family contrast identifier |
| citation audit tables | Citation edges or pooled summaries according to file; repeated rows are not additional runs | see column catalogue |
| figure CSVs | The exact plotted records or summaries | see figure_manifest.json |
| recovery/attempt-index.csv | One archived snapshot, which may be a checkpoint backup | campaign_id + corpus + run_id + archive_snapshot_id |

## Identifiers and joins

Run IDs are text; preserve leading zeros. Join main outputs by run_id with campaign
and corpus validation. Never join independent populations just by seed. Node
occurrence IDs/indices preserve repeated names; names alone are not unique keys.
Citation doc_ids (`rep_...`, `fed_...`) are distinct from PDF filename numbers and
bibliography keys. Resolve through the corresponding corpus manifest; do not renumber.

## Units and populations

- N is the planned number of sequential evidence batches/generation steps.
- Seed/repetition labels are recorded configuration, not proof of matched random
  draws across providers. k_doc scales the batch retrieval budget; it is not a
  per-document guarantee of unique retained chunks.
- Root depth is 0; XML wrapper elements do not increase feature-tree depth.
- Similarities are cosine scores. Threshold is inclusive (`>=`); primary tau is
  0.4. Threshold and policy must be retained when comparing rescore records.
- Ratios generally lie in [0,1]; explicitly named percent fields use [0,100].
  Token counts are provider/estimator-specific quantities, not word counts.
  Recorded wall-time fields ending in seconds are seconds; retain their original
  definitions and do not sum duplicated recovery checkpoints as new expenditure.
- Semantic scoring targets completed, extractable finals. Failed runs retain their
  place in planned-run denominators; their last valid checkpoints are not substituted.
- Strict-admissibility results are a separate sensitivity population. SAT,
  FeatureIDE-reader acceptance, local structural conformance and completion differ.
- Aggregate wide.csv does not automatically incorporate later FeatureIDE,
  dead-feature-count or sibling sidecars; use those explicitly and document joins.

## Matching, reach and uncertainty

Independent-max permits many-to-one credit. One-to-one and other policy variants
have their own labels in the rescore outputs; do not interchange them. F1 uses the
declared precision/recall definition, not arbitrary reach recall substitution.
Reach is the ancestor closure of directly attested features for the specified
document set. `inputs/calibration/rho.json` is historical calibration, not authority
for replacing the closed-reach values in semantic-current-v3.

Citation referential integrity establishes valid identifiers, not claim support.
Reference attribution agreement is not an expert entailment judgment. Document
coverage does not prove meaningful use of every cited source. Parent-label matching
and sibling agreement capture different structural properties.

Blank/null/NaN values mean unavailable as indicated by companion status/reason,
not zero. Statuses include ok, ineligible, not_applicable, unsupported,
missing_artifact and evaluator_error. Never impute failed-run semantic scores to
zero without labelling a separate estimand. Conditional-success statistics need
their eligible n alongside planned n. Bootstrap intervals, multiplicity families
and minimum-n policies are documented in archived family specs/results; a
nonsignificant contrast does not establish equivalence or saturation.

## Release-derived metadata

artifact-manifest.json: original files carry `source`, current `path`, previous
archive path, byte length and SHA-256. `derived_files` contains preparation outputs.
The manifest cannot include its own checksum; checksums.sha256 includes it.
Bundle-level SHA256SUMS separately covers compressed archives and bundle metadata.
attempt-index.csv preserves UTC timestamps when recorded, zero-based failed_step,
current outcome, explicit linkage basis and recovery history JSON. Missing metadata
is blank; archive UUID order does not establish call chronology.
