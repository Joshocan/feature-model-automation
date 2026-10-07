# Manuscript-to-archive reconciliation

Status: pending the exact final manuscript source or PDF and version identifier.
The release-support campaign-outcomes.csv is not proof that every paper claim
has been checked. Do not treat older reports as the final manuscript.

For every table, figure and quantitative claim, record:

| Manuscript location | Source artifact + SHA256 | Population/filter | Formula/statistic | Rounding | Recomputed value | Reported value | Verdict |
|---|---|---|---|---|---|---|---|
| Pending final manuscript | | | | | | | |

1. Freeze the manuscript version and use the archive's selected analysis versions,
   not similarly named older directories in the software workspace.
2. Recalculate all table cells from their saved inputs; verify corpus, model, arm,
   N, seed selection, matching policy, threshold and scoring population.
3. Check denominators: planned vs completed vs admissible; missing is not zero.
   Keep primary extractable-completed and strict sensitivity populations distinct.
4. Check precision/recall/F1 policy, ancestor-closed reach, rho, citation integrity
   versus evidence support, and label-based parent agreement versus sibling agreement.
5. Check statistical family membership, adjustment, effect sizes and intervals.
   Avoid claiming equivalence from a nonsignificant result or a numerical saturation
   point without its declared criterion.
6. Verify figure captions against saved source data and report post-pilot decisions
   and historical prompt limitations. Exclude uncollected expert findings.
7. Record discrepancies and corrections, then author sign-off against the final
   manuscript and archive hashes. Refresh the release if either changes.

Reviewer, date, manuscript identifier and sign-off: pending.
