# Software freeze and clean-environment reproduction

Status: NOT PERFORMED by release-document preparation.

A successful file checksum test establishes integrity, not computational
reproducibility. A source HEAD identifier alone does not cover uncommitted changes
used for evaluation. See analysis-lineage.json for recorded implementation hashes.

## Required procedure

1. Resolve the maintained software changes, run its tests, pin dependencies and
   the supported Python/Java environment, and record the evaluator/encoder and
   FeatureIDE versions and hashes. Inspect licensing and secrets before committing.
2. Create an approved software commit and release tag. Record the full commit,
   tag, source-archive checksum and software DOI when available. This document
   does not authorise or claim an automatic commit, push or release.
3. In a clean checkout and fresh environment, install only declared dependencies.
   Download the candidate data into a separate directory and verify all checksums.
   Use artifact-manifest.json source/path mappings where evaluators expect original
   software-relative paths; document the restoration command or loader explicitly.
4. Reproduce evaluation from the archived XML and reference inputs, followed by
   aggregate tables, statistical comparisons and figures, writing to a NEW output
   directory. Use the pinned local encoder and recorded policy configuration.
   Do not make generation API calls or silently reuse old evaluation outputs.
5. Compare run sets, missingness/status, integer counts and numerical metrics.
   Document justified numeric tolerances and any platform-dependent rendering
   differences in advance. A PDF byte difference alone is not a metric mismatch.
6. Save environment versions, exact commands, logs, comparison results, output
   hashes and reviewer sign-off. Resolve discrepancies before publication.

## Reproduction record

Software commit/tag: pending. Environment lock: pending. Data manifest hash tested:
pending. Commands/logs: pending. Differences/tolerances: pending. Reviewer/date:
pending. Result: pending.

Full model generation is a separate, paid, stochastic workflow requiring corpus
access and providers. It is not required to test recomputation of the archived
results, and no claim of byte-identical new LLM responses should be made.
