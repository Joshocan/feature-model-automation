# Analysis guide

The main campaign has 694 planned and 500 completed outputs. Pilots, retries and
the separate Astra extension are not extra main-campaign repetitions.

| Directory | Purpose and primary files |
| --- | --- |
| `inventory/` | Planned runs, artifact identities and recorded outcomes |
| `aggregate/` | Joined metrics (`wide.csv`) and configuration summaries |
| `structural/`, `semantic/`, `provenance/` | Original evaluator results and diagnostics; structural sidecars include SAT, FeatureIDE and sibling agreement |
| `matching-sensitivity/` | Independent-max versus one-to-one scores, threshold sweeps and rankings |
| `statistical-tests/` | Declared hypothesis families and strict-population sensitivities |
| `negative-control/` | Own and other-corpus reference scores for all completed runs, thresholds 0.3–0.8, both policies |
| `attribution-agreement/` | Consistent citation-weighted baseline, restricted alignments and comparison with earlier published baseline values |
| `reference-partitions/` | Run-level and mean recall over attested, organising and ancestor-closed reach partitions |
| `logical-validity/` | SAT eligibility, unsatisfiable-model diagnoses and W-condition checks |
| `conformance-denominators/` | Completed/planned denominators and their different percentage-point contrasts |
| `verification/` | Input hashes, source-search evidence and the historical analysis report |

## Current interpretation

For attribution agreement, use `attribution-agreement/table3.csv`. The baseline
averages |attr(alpha(f))| / |D| over the **same applicable citations** as agreement.
The comparison with the earlier paper values is retained to explain the corrected
weighting, not as an alternative final table. Pooled citation differences and
equal-run mean differences are distinct statistics. Restricted alignment results
use similarity ≥0.8 or unique exact-name matching.

For reference partitions, the denominators are Repair: 58 attested, 42 organising,
84 reach; Federation: 36 attested, 94 organising, 59 reach. Reach includes ancestors.
Keep threshold and matching policy when joining results. Run means and confidence
intervals use whole-run resampling as specified in the saved outputs.

For conformance, conditional yield is conformant/completed and unconditional yield
is conformant/planned. The published Delta-Conf examples reproduce differences
in conditional yield. If no outputs completed, conditional conformance is undefined.

The historical report in `verification/analysis-history.md` starts with an earlier
stopped baseline check and then records its resolution. Read the final follow-up
and `verification/status.json`; do not treat the opening stopped attempt as the
current status. No failed or superseded metric has been silently erased.
