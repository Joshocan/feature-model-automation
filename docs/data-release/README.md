# Feature-model research data: automated experiments

This local release candidate accompanies *Conformance Is Not Correctness:
Evaluating LLM-Constructed Feature Models*. It contains automated campaign and
pilot evidence, not expert ratings or expert-analysis results. It is not yet
approved for publication; see RELEASE_CHECKLIST.md and release-validation.json.

## Contents and populations

- `experiments/main-campaign/`: the 694 planned run records, including 500
  completed final XMLs and incomplete outcomes. Completion is not conformance.
- `experiments/pilots/`: characterisation runs, including the four isolated
  schema-depth-v2 directories. Do not pool them into main-campaign repetitions.
- `experiments/expert-astra-extension/`: separate Astra N=1 generated outputs.
  These are generation evidence, not human-evaluation results.
- `inputs/`: reference XMLs, attribution, feature partitions, historical
  calibration, document manifests, document orderings and encoder metadata.
- `protocol/`: generation/evaluation snapshots, documentation, deviations,
  analysis lineage and machine-readable CSV column catalogues.
- `analysis/`: inventory, structural, semantic, provenance, matching-sensitivity,
  statistical and aggregate outputs, with original version labels preserved.
- `recovery/`: original archived snapshots and a derived attempt index.
- `paper/`: saved figures with source data, plus an outcome table and table index.

Expert-study placeholders and local preparation backups are excluded from bundles.
No expert conclusions should be attributed to this release.

## Verification

After obtaining bundles, verify `SHA256SUMS` from the bundle directory, extract
all five archives into the same NEW empty directory, then run:

```sh
shasum -a 256 -c checksums.sha256
```

Do not extract over active experiment data. `artifact-manifest.json` maps archived
`path` values to original software-relative `source` paths and records original
hashes. Derived release documentation is listed separately. Internal historical
paths, hashes and unresolved labels have not been rewritten.

## Reproduction scope

Tables/figures can be regenerated from saved metrics using the companion software.
Evaluation can be rerun from final XMLs, reference inputs and the pinned local
encoder, without generation APIs. Full generation requires source-text access,
index construction, provider access and spending; fresh stochastic outputs are
not promised to reproduce identical bytes.

The final compatible software release/DOI is pending. The original working tree
was dirty: its HEAD is not a sufficient identifier of the evaluator. See
protocol/analysis-lineage.json for recorded implementation and input hashes.
This preparation has not performed a clean-environment scientific recomputation.

## Restrictions and release timing

Original contributions are licensed under CC BY 4.0; see LICENSE.md and
LICENSES/README.md for scope and third-party exclusions. Copyright 2026 Joshua
Tetteh Ocansey, Yngve Lamo, Adrian Rutle, Fazle Rabbi. This licence choice does not
constitute publication approval: see RELEASE_APPROVAL.md for the pending gates.
Source-paper PDFs, processed chunks and vector
indexes are omitted pending redistribution review. The corpus manifests retain
document identities and source links. Generated descriptions may quote papers;
omitting PDFs does not automatically clear every output for redistribution.
Personal absolute paths are preserved in some historical records; the pattern-scan
report identifies their files without changing evidence. No automated scan is a
complete security or privacy review. Public outputs can compromise ongoing expert
blinding even when the explicit mapping key is excluded. Author approval is required
before publishing. No upload, Git commit or push is performed by the release tools.
