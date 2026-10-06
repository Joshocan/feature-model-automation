# Protocol adaptations and evaluation clarifications

This is a retrospective consolidation, not a claim of preregistration. Frozen
historical artifacts are preserved unchanged. See documentation/evaluation-decisions.md,
documentation/analysis-followup-plan.md and configuration-snapshots/ for evidence.

| Topic | Archive interpretation |
| --- | --- |
| Pilot model/prompt/budget changes | Pilot-only configurations remain separate; original run metadata identify each setting. Do not describe all pilots as identically configured. |
| Main model participation | DeepSeek Flash and GLM have the open-weight matrix; Astra has the reduced frozen matrix. Do not imply a balanced three-model design. |
| Sampling settings | Main open-weight settings use temperature 0.2 / low reasoning; Astra records 1.0 / medium. No claim of equivalent internal compute or controlled cross-provider variance. |
| Output allowance | Main metadata record 32,768 tokens. Earlier pilot caps remain historical facts. Equal caps do not imply equal effective reasoning/output allocation. |
| Failure/recovery | Incomplete outcomes remain in the inventory. Archived attempts and current outcomes are linked in recovery/attempt-index.csv. Checkpoint snapshots are not additional repetitions. |
| Semantic eligibility (D01) | Author decision recorded 2026-09-29: completed extractable finals primary; strict-admissible finals sensitivity. This follows generation. |
| Structural scope (D02) | Local schema/W checks, SAT and FeatureIDE-reader acceptance are distinct. FeatureIDE sidecar records pinned parser results; older base summaries still say unsupported. Do not silently relabel the base aggregate. Final strict-gate approval remains explicit. |
| Encoder (D03) | semantic-current-v3 records model revision, weight/tokenizer hashes and runtime versions. Historical pending labels are retained; clean-environment reproduction remains a separate gate. |
| Inferential families (D04) | Explicit A/B/C family specifications/results are archived; declarations follow descriptive inspection. Do not call them prospective preregistration. No unapproved equivalence margin or numeric saturation estimator is introduced. |
| Attribution and reach | Current provenance uses corpus manifests; current semantic reach closes attested features upward. Earlier direct-attestation calibration is not the same quantity. |
| Matching sensitivity | Keep independent-max and one-to-one results separately; label tau and policy. No threshold switch based on favourable outcomes is asserted. |
| Hierarchy | Parent-match is reference-parent-label agreement; sibling agreement is a separate label-independent grouping diagnostic with its own matching policies. |
| Expert assessment (D05) | Excluded from this release scope; neither pending responses nor expert correctness conclusions are supplied. |

The source snapshots contain historical absolute paths. Reorganisation changed
archive paths only, with manifest mappings; it did not repair generated models,
modify original metrics, or rerun calls. release-validation.json and
analysis-lineage.json record technical checks, not blanket scientific approval.
Any absent pilot prompt version must be disclosed using the prompt coverage audit.
