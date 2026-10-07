# Feature Model Automation

Replication software for **Conformance Is Not Correctness: Evaluating
LLM-Constructed Feature Models**. The Python package is named `fame`.

The current pipeline uses one N-batch generation engine, RAG or Non-RAG grounding,
and optional metamodel guidance. It does not use the retired SS/IS four-pipeline
design or LangChain orchestration.

## Release status

This is a release candidate, not a certified reproduction release. See
[release checklist](docs/software-release-checklist.md) for unresolved gates.
The final version and DOI must be settled before publication.

## Licence

The software is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 Joshua Tetteh Ocansey, Yngve Lamo, Adrian Rutle, Fazle Rabbi.
The separate research-data archive remains CC BY 4.0; its notices and templates
under `docs/data-release/` retain that scope. Third-party material retains its
own notices and licences and is not relicensed by this software licence.

## Start here

1. [Install and test](SETUP.md).
2. [Restore data and reproduce results](docs/reproduce.md).
3. Read the [experiment protocol](IFS_2027_EXPERIMENT.md),
   [evaluation specification](docs/evaluation-spec.md) and
   [metric dictionary](docs/metric-dictionary.md).

## Maintained entry points

| Task | Scripts |
|---|---|
| Corpus preparation (requires source papers) | `build_chunks.py`, `build_index.py` |
| Generation (paid provider calls) | `build_run_matrix.py`, `campaign.py`, `smoke_run.py` |
| Inventory and conformance | `inventory_campaign.py`, `evaluate_structure.py`, `evaluate_featureide.py` |
| Semantics and provenance | `evaluate_semantic.py`, `evaluate_provenance.py` |
| Matching/hierarchy sensitivity | `tau_rescore.py`, `compare_matching.py`, `evaluate_siblings.py` |
| Aggregation/statistics | `aggregate_campaign.py`, `analyse_family.py`, `verify_evaluation_outputs.py` |
| Paper figures | `plot_paper_figures.py` |
| Data verification/restoration | `restore_research_data.py` |

Scripts are under `scripts/`; run them from the repository root. Use `--help`
for current arguments. Generation is never necessary to recompute metrics from
saved XML. Never launch generation as an installation or reproduction smoke test.

## Data and scope

Large outputs belong in the separate `feature-model-research-data` archive.
The main campaign contains 694 planned runs and 500 completed outputs; pilots,
recovery snapshots and the Astra N=1 expert extension are separate populations.
Completion, structural conformance, FeatureIDE acceptance and semantic correctness
are different outcomes. Primary semantic comparisons use completed extractable
outputs; strict admissibility is a separate sensitivity analysis.

`results/`, credentials, processed corpus text, vector indexes and local environments
are excluded from Git. Do not publish a ZIP of the entire working directory.
Do not place returned expert forms or identity keys in the software repository.

The expert packet generator supports historical and current citation layouts,
but the tracked form retains extra source-table rows. It must be reviewed and
prepared as a separate blank copy before generating new packets; it is not a
returned-response extractor. Expert-response analysis remains out of this release's scope.

## Reproducibility limits

Use the recorded encoder revision and versions, not the latest embedding model.
FeatureIDE has a separately pinned JAR; see [its guide](docs/featureide-evaluation.md).
Source texts and indexes are needed only to rebuild generation inputs and may
require separate access rights. New stochastic LLM runs are not promised to
produce identical bytes. A clean test suite alone does not verify every paper table.
