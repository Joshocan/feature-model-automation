# Metric dictionary — evaluation 0.1.0

Companion to [evaluation-spec.md](evaluation-spec.md). `null` always carries a
status/reason. The run is the replication unit; per-node counts are within-run data.
Rules labelled proposed are not frozen until the decision register is resolved.

| Metric | Definition / denominator | Eligibility | Question |
| --- | --- | --- | --- |
| Completion rate | Runs with all planned steps accepted and final artifact / planned runs | Entire matrix; missing status shown | H1a, reproducibility |
| Parseability | Raw final XML parsed without repair | Final response available | Conformance |
| XSD validity | Validation against frozen schema, independently recorded | Parseable XML | Conformance, H3 |
| W1 | Exactly one feature root, with the configured root name | Extractable structure; otherwise not evaluable | Conformance |
| W2 | All structural feature names unique | Extractable structure | Conformance |
| W3 | Plain requires/excludes constraints have two operands | Parsed constraints; encoding interpretation pending D02 | Conformance |
| W4 | Every constraint variable resolves to a structural feature | Parsed structure and constraints | Conformance |
| W5 | Abstract rule separates plain binary relations from other formula constraints | Schema translation pending D02; never reject canonical XML formulas merely for being formulas | Conformance |
| Structural conformance | XSD, W1–W5 and stated tree invariants all pass | No unknown required checks; unavailable is not a pass | H3, title |
| FeatureIDE parse | Acceptance by an actual pinned FeatureIDE parser | Tool installed and supported; not inferred from XSD | Tool compatibility |
| Output-contract compliance | Frozen declaration/envelope, identifier, description and trace requirements, separate violation fields | Raw final output and structure; sentence support is not a syntax check | H3 |
| Strict admissible yield | Completed AND XSD, W1–W5, tree/name/envelope checks, complete per-feature traces and L1 integrity / planned runs | Operational sensitivity gate pending D02 final approval; separate historical pilot gate | H3, reproducibility |
| SAT | At least one valid product under encoded tree and constraints | Conforming supported model; unsupported operators cannot be skipped | Logical correctness |
| Dead-feature ratio | Non-root features never selectable / non-root features | SAT model, supported encoding; unsatisfiable models reported separately | Logical correctness |
| Constraint counts | Number of requires, excludes and other rules, including zero | Parsed supported rules; D02 classification | Variability |
| Feature count | Named structural node occurrences, including root and abstract nodes | Extractable structure | Shape, redundancy |
| Maximum depth | Maximum feature-edge distance from root; root=0 | Single rooted feature tree | Shape |
| Mean branching | Sum of immediate feature children / number of feature nodes with children | Extractable tree; null if no internal nodes | Shape |
| Group proportions | Internal and/or/alt counts / total internal nodes | Extractable tree; invalid empty groups flagged, not treated as ordinary internal nodes | Variability |
| Mandatory ratio | Non-root nodes explicitly mandatory=true / non-root nodes | Extractable structure; not claimed to equal logical core-feature ratio | Variability |
| Degenerate flag | No or groups, no alt groups, no non-root mandatory=true | Valid structure; ignores cross-tree constraints by definition | Variability, H3 |
| Non-degenerate yield | Strict-admissible AND not degenerate / planned runs; also conditional share | Strict sensitivity gate pending D02 final approval | Reliability |
| P / R_total / F1_total | Independent-max threshold counts and harmonic mean, as defined in specification | Primary: completed final with extractable feature tree; strict-admissible sensitivity; never substitute checkpoints | H1b, H2, H3 |
| R_reach / R_attested / R_organising | Matched reference members / reference subset size | Same semantic population; subset IDs must resolve and be unique | Reference coverage |
| rho | Attributed/reachable reference count / all reference features, with subset definition recorded | Reference and attribution integrity | Calibration, not a proven semantic ceiling |
| Parent-match | Correct mapped parent links / evaluable mapped links | Matched non-root child and matched parent; n=0 gives null | Structural correctness |
| Group-kind agreement | Correct group-kind pairs / aligned pairs internal in both trees; publish 3x3 confusion counts | Deterministic above-tau alignment; count leaf/internal mismatches separately | Variability correctness |
| Mandatory agreement | Equal explicit mandatory flags / aligned non-root pairs | Exclude pairs mapping to reference root; show both error directions | Variability correctness |
| Exact duplicate surplus | Sum over names of max(count-1,0) / all feature occurrences | Extractable structure, including nonconforming diagnostics | Redundancy |
| Near-duplicate pair counts | Unordered generated occurrence pairs with different exact names and cosine >= 0.8 or >= 0.9 | >=2 nodes; exact-name duplicate surplus is reported separately; counts are not rates | Redundancy sensitivity |
| Trace coverage / L0 | Features with exactly one description and a valid nonempty final Trace suffix / all features | Extractable structure, including uncited features | Provenance |
| L1 integrity | Cited feature-document pairs whose IDs exist in corpus manifest / distinct cited pairs | Parseable IDs; empty denominator null; malformed traces separate | Provenance |
| Current/cumulative exposure | Citation pairs with recorded current/prior evidence exposure / pairs with known lineage and exposure | Track pair birth; retained citations not required in current batch | Provenance, not entailment |
| Attribution agreement | Supported generated-reference-document triples / evaluable cited pairs under selected feature alignment | Matched features and annotated reference attribution; unannotated cases separately counted | Reference-relative provenance |
| Document citation coverage | Distinct real cited documents / corpus documents; additionally / documents actually exposed | Report both denominators; does not measure evidence use or correctness directly | Provenance |
| Recency | Citation-pair introduction step minus earliest actual evidence exposure step | Multiple steps and known lineage; show exposure baseline; N=1 not applicable | Provenance |
| Input margin | Configured context allowance minus assembled input and required reserved output, under recorded provider accounting | Attempted step and known token convention; distinguish estimates from actual usage | H1a |
| Output exhaustion | Recorded length termination; tokens/headroom separately | Attempted calls; headroom is observed, not required future size | H1a |
| Cost/time | Per run/step reported usage and wall time; separate total recovery expenditure and selected attempt | Known usage/pricing; unknown is not zero | Efficiency |
| Trajectory | Same metrics on valid checkpoints, labelled by step and corpus fraction processed | Diagnostic cohort with at-risk counts; never replace failed final scores | H2/H4 exploratory |

The degeneracy definition is inherited as a structural proxy. A flagged model may
still have meaningful cross-tree constraints; do not describe every flagged model
as logically variability-free. Exact duplicates, near similarity, unmatched
reference features, and unsupported-in-corpus features are different concepts.
Only a documented evidence audit can substantiate the last claim.
