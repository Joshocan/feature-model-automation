# Conformance Is Not Correctness: research data

This archive accompanies *Conformance Is Not Correctness: Evaluating
LLM-Constructed Feature Models*. It contains saved generation evidence,
automated evaluations, anonymised expert assessments, figure data and
reproduction evidence. It does not contain private expert workbooks, participant
identity keys, credentials, source-paper PDFs or retrieval indexes.

The data record has reserved DOI **10.5281/zenodo.23206866**. The companion
software's published record is **10.5281/zenodo.23210345**. The additional public
expert-input and archive-restoration functionality requires the updated software
source described in `protocol/reproduction/README.md`; the older software release
alone does not contain it. Publication approval and the final compatible software
tag are separate decisions recorded in `RELEASE_APPROVAL.md`.

## Start here

| Question | Start with |
| --- | --- |
| Which runs were planned and completed? | `analysis/inventory/inventory-current-v1/runs.csv` |
| Where are the joined automated measurements? | `analysis/aggregate/aggregate-current-v4/wide.csv` |
| How do matching policies change scores and rankings? | `analysis/matching-sensitivity/matching-comparison-v2/` |
| How strong is agreement against the wrong reference? | `analysis/negative-control/` |
| What is the citation-weighted attribution baseline? | `analysis/attribution-agreement/` |
| What are recall over reference partitions and logical failure causes? | `analysis/reference-partitions/` and `analysis/logical-validity/` |
| What denominators produce the conformance differences? | `analysis/conformance-denominators/` |
| What did the experts assess and report? | `expert-study/README.md` |
| Which figures should I use? | `paper/figures/publication/` |
| How can I reproduce or verify the evidence? | `protocol/reproduction/README.md` |

## Populations and directory layout

- `experiments/main-campaign/`: 694 planned runs, including 500 completed final
  XMLs. Incomplete outcomes remain part of planned-run denominators. Completion,
  conformance, satisfiability and non-degeneracy are different outcomes.
- `experiments/pilots/`: historical model-selection and capability evidence,
  including the four isolated schema-depth-v2 pilots. Do not pool these with the
  main campaign. Folder names are descriptive; original campaign identities and
  dates remain in metadata and the manifest.
- `experiments/expert-astra-extension/`: ten separate Astra N=1 RAG outputs,
  five per corpus; two are included in the ten-output expert sample.
- `inputs/`: corpus manifests, reference XMLs, attribution, feature partitions
  and calibration records. Reference attribution is not an expert judgement.
- `protocol/`: frozen configurations, prompts, decision records, lineage and
  reproduction evidence. Historical snapshots are evidence, not silently updated
  declarations of the final protocol.
- `analysis/`: automated run-level scores and derived comparisons.
- `expert-study/`: anonymised response inputs, selection, study definitions and
  analysis outputs. Raters are identified only as Expert A and Expert B.
- `recovery/`: archived attempts and their links to current outcomes. Retries are
  not additional independent repetitions.
- `paper/`: figure files and data, table-source index and manuscript reconciliation.

## Verify and restore

Use the research-data ZIP, not a GitHub source-code download: this preparation
repository intentionally ignores several large payload directories in Git. The
packaging tool selects files from the verified archive catalogue, independently
of Git tracking. A Git clone alone is not the complete research dataset.

Verify the downloaded ZIP against its accompanying `SHA256SUMS` if supplied.
Extract into a **new directory**, then run inside the extracted archive:

```sh
shasum -a 256 -c checksums.sha256
```

Install the compatible software environment as described in its README. From
that software checkout, first verify and preview restoration:

```sh
./.venv-release/bin/python scripts/restore_research_data.py \
  --archive /absolute/path/to/feature-model-research-data \
  --destination /absolute/path/to/feature-model-automation
```

Repeat with `--apply` to restore absent data files. Differing existing files cause
failure before copying. The tool maps descriptive archive names back to the
historical paths expected by the scripts. `restore_sources` lists additional
paths backed by identical bytes, avoiding duplicate archived datasets. Always
use the manifest rather than guessing locations from old paths embedded in files.

No generation calls are needed to analyse saved results. Recomputing embeddings
requires the documented cached encoder. Regenerating the original campaigns is
a separate workflow requiring paper access, indexes and provider expenditure;
identical stochastic outputs are not guaranteed.

## Reading and citing results

Use `DATA_DICTIONARY.md`, the README beside each analysis and
`paper/tables/README.md`. Preserve run IDs as text. Keep corpus, matching policy,
threshold and eligibility in joins. Missing values are not zero. Do not treat
the small expert sample as a population-wide model ranking.

Public-facing filenames omit preparation dates and the label "reanalysis".
Original source paths, execution timestamps and hashes are retained inside
provenance records. `protocol/archive-path-map.csv` records renamed existing
files. Original evidence is not altered merely to modernise terminology.

Eligible original contributions are CC BY 4.0. Copyright 2026 Joshua Tetteh
Ocansey, Yngve Lamo, Adrian Rutle and Fazle Rabbi. Consult `LICENSES/README.md`
for third-party exclusions. Privacy/rights review, acceptance of the historical
pilot-prompt limitation and final manuscript reconciliation remain researcher
decisions; copying these materials does not grant publication approval.
