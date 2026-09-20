# Feature Model Experiment: Phased Implementation Workflow

This workflow implements the design in
`data/ground_truth/implementation-brief.md`. It replaces the former four-pipeline
(`SS/IS × RAG/Non-RAG`) workflow. Single-stage and document-incremental generation
are now endpoints of one granularity parameter, `N`.

The implementation brief is the design authority. This document is the delivery
sequence: what changes, what is removed, what must exist at each gate, and when
experiments may start.

## Non-negotiable rules

1. Generation is one loop parameterised by corpus, ordering, `N`, grounding
   (`rag` or `nonrag`), model, prompt variant, repetition, and seed.
2. Every document is consumed exactly once at every `N`. Batches are contiguous
   slices of the frozen ordering.
3. RAG and Non-RAG resolve text from the same canonical `chunks.jsonl`.
4. Evaluation data, reference models, and scores never enter generation.
5. Prompts, metamodel, queries, orderings, models, and encoders are frozen and hashed
   before the measured campaign.
6. An over-limit prompt is recorded as infeasible, never silently truncated.
7. Logs are written during execution; missing D7–D11 or D22 data is not reconstructed.
8. Do not select `N`, `k`, tau, alpha, beta, or structural weights from outcome scores.

## Target repository shape

Names may change, but the final responsibilities must stay separate and there must
be only one generation engine.

```text
config/
  experiment.yaml
data/
  corpora/{repair,federation}/
  manifests/{repair,federation}.csv
  processed/{repair,federation}/chunks.jsonl
  ground_truth/{repair,federation}/
    fm_ref.xml
    attribution.csv
    feature_partition.csv
  orderings.json
fame/
  ingestion/
  retrieval/
  generation/                      # single N-step loop
  logging/
  evaluation/                      # derived computations only
prompts/
results/<campaign_id>/             # immutable outputs for one campaign
scripts/
  prepare_corpus.py
  build_index.py
  validate_retrieval.py
  smoke_run.py
  run_campaign.py
  derive_metrics.py
  analyse_campaign.py
tests/
```

## Phase 0 — Protect the new baseline

### Work

- **0.1** Create a named backup, tag, or archive of the current repository before deletion (e.g. `git tag pre-refactor-2026-09-20`).
- **0.2** Give every existing item one disposition: `keep`, `migrate`, `regenerate`, or `remove`. Confirm the disposition of ambiguous trees: `fame_web/`, `notebooks/`, `clear/`, `ss_k_ablations/`, `results/MDE_intelligence_2026/`.
- **0.3** Preserve raw papers, the implementation brief, reference sources, inputs needed to reconstruct D1–D4/D21, and irreplaceable expert annotations.
- **0.4** Record checksums for preserved raw inputs.
- **0.5** Exclude credentials, environments, caches, and Chroma data from the archive.

### Exit gate

- The pre-rewrite state is recoverable outside the active experiment tree.
- Every Phase 1 deletion has an explicit disposition and replacement, if needed.

## Phase 1 — Clean up the superseded experiment

This phase removes the old experiment from the active tree. It has two passes so
reusable implementation details can be migrated before old wrappers disappear.

### Pass A: remove derived and stale artefacts

Safe deletions — no code dependency. Do these immediately.

- **1A.1** Old chunk stores: `data/processed/repair/`, `data/processed/federation/`. Chunks at 6,000 chars are void; every index based on them is invalid.
- **1A.2** All Chroma persistence: `data/chroma_db*/`. The new design has exactly one collection per corpus, derived from the new `chunks.jsonl`.
- **1A.3** Four-pipeline analysis output: `results/analysis/overall_four_pipelines/`, `results/rag/`, `results/non_rag/`, `results/ground_truth/export/`. Regenerate exports from frozen D3/D4/D21.
- **1A.4** SS-RAG k-ablation tree: `ss_k_ablations/` (including `heqed/` subtree, all `fig_*.png`, `table_*.csv`, `ablation_*.json`).
- **1A.5** Workshop archives that must not mix with the new campaign: `results/MDE_intelligence_2026/`. If kept for historical reporting, move outside active `results/`.
- **1A.6** Redundant manifests and working spreadsheets: `data/raw/federation/manifest.csv`, `data/raw/repair/manifest.csv`, `data/raw/repair/manifest_fed_corrected.numbers`, `data/raw/repair/manifest_rep_corrected.numbers`. `manifest_fed.csv` and `manifest_repair.csv` are canonical.
- **1A.7** Duplicate or unrelated ground-truth files: `data/ground_truth/repair_feature_model.xml` (duplicate of `repair.xml`); `data/ground_truth/REAL-FM-{3,15,16,17,19}.xml` and `REAL-FM-16_problems.log` if not inputs to the new D3/D4/D21.
- **1A.8** Superseded prompts: `prompts/single-staged_non-rag-template.txt`, `prompts/iterative_refinement_prompt_featureide_xml_v2.txt`, `prompts/fm_extraction_and_merge_prompt.txt`, `prompts/heqed_lens.txt`, `prompts/core_structure_metamodel_extraction.txt`, `prompts/core_structure_metamodel_extraction_iterated.txt`.
- **1A.9** `prompts/specifications/old/` after confirming none of its files is an input to the new frozen D5/D6 artefacts.
- **1A.10** Caches, logs, temporary files, `.DS_Store`, and generated images or reports.

### Pass B: replace and remove legacy implementation

Only remove once replacement tests pass in Phase 5. Migrate reusable primitives — PDF loading, model clients, XML parsing, conformance checks, SAT checks, and embedding adapters — first.

- **1B.1** Legacy pipeline modules: `fame/rag/ss_pipeline.py`, `fame/rag/is_pipeline.py`, `fame/nonrag/ss_pipeline.py`, `fame/nonrag/is_pipeline.py`, plus `fame/nonrag/{cli_utils,prompt_utils,prompting,llm_ollama_http}.py` after their contents migrate.
- **1B.2** Pipeline entry-point scripts: `scripts/run_ss_rag.py`, `scripts/run_is_rag.py`, `scripts/run_ss_nonrag.py`, `scripts/run_is_nonrag.py`, `scripts/run_fame.py`.
- **1B.3** Old SS-RAG `k` ablation: `scripts/ablate_ss_rag_k.py`.
- **1B.4** Four-pipeline aggregation/comparison: `scripts/build_overall_pipeline_data.py`, `scripts/compare_overall_vs_topk.py`, `scripts/aggregate_paper2_results.py`, `scripts/overall_eval_plot_utils.py`, `scripts/evaluate_selection_baselines.py`, `scripts/run_paper2_experiments.py`.
- **1B.5** Proxy-selector arm: `fame/evaluation/proxy_baselines.py`, `proxy_compare.py`, `proxy_config.py`, `proxy_consensus.py`, `proxy_evidence.py`, `proxy_reporting.py`, `proxy_selector.py`; and `scripts/run_proxy_ablation.py`, `rank_proxy_fm.py`, `compare_proxy_vs_gt.py`.
- **1B.6** Four-pipeline plots that cannot consume the new run schema: `scripts/plot_eval_rag_vs_nonrag.py`, `plot_eval_iteration.py`, `plot_eval_overall_capability.py`, `plot_eval_model_comparison.py`, `plot_eval_validity.py`. Rewrite against the N-axis in Phase 9.
- **1B.7** SS/IS-specific tests, after the unified engine covers equivalent behaviour: `tests/test_ss_nonrag.py`, `test_is_nonrag.py`, `test_ss_rag.py`, `test_is_rag.py`.
- **1B.8** Config fields to remove from `config/fame.yaml`: `ss_rgfm`, `is_rgfm`, `ss_nonrag`, `is_nonrag`, `per_source`, 6,000-character limits, `paper2_experiments.yaml`, and old result-directory conventions. Also remove the legacy `model_max_tokens` entries for retired models (`gpt-4.1`, `gemini-3.1-pro-preview`, `gpt-oss:120b-cloud`, `deepseek-v3.2:cloud`, `glm-4.7:cloud`).
- **1B.9** Duplicate manifests and reference models after one canonical D1 and D3 per corpus has been verified.
- **1B.10** Rewrite `README.md`, `SETUP.md`, `IFS_2027_EXPERIMENT.md`, and any related guides so no active command or explanation presents SS and IS as separate pipelines.

### Keep and adapt

- `fame/ingestion/`, adding complete front/back-matter stripping, stable `doc_id`,
  offsets, overlap, and one JSONL serializer.
- `fame/vectorization/` and `fame/retrieval/`, enforcing `search_document:` and
  `search_query:` prefixes, two corpus collections, four fixed sub-queries, `doc_id`
  filters, and `chunk_id` deduplication.
- evaluation primitives for conformance, FeatureIDE parsing, satisfiability, semantic
  similarity, tree structure, and provenance. Evaluation must never reach generation.
- provider clients and rate-limit handling, updated to return token counts, model
  version, finish reason, and context-window information.

### Exit gate

- Derived artefacts in Pass A are gone and the Pass B removal list is locked. Pass B
  closes in Phase 5 as soon as the unified engine passes its replacement tests.
- No generated result or old index remains in an active input directory.
- Preserved primitives have tests or a recorded migration target.

## Phase 2 — Resolve open decisions and freeze the protocol

### Blocking decisions

- **2.1** Choose exact chunk target/max size within 1,200–1,500 characters and exact overlap within 10–15 percent.
- **2.2** Define the literal provenance marker grammar and its parser.
- **2.3** Verify model identifiers, versions, context windows, output caps, temperature or reasoning settings, and serving endpoints.
- **2.4** Keep `k_doc` provisional until Phase 4 measures chunks per document.

### Non-blocking analysis decisions

- **2.5** Fix retained-candidate count `f_m`.
- **2.6** Fix the SAT solver name and version.
- **2.7** Fix the expert rubric (start early; raters are the long pole).

### Freeze D5, D6, and D12

- **2.8** D5: `metamodel.ecore`, literal metamodel prompt block, and hashes.
- **2.9** D6: extraction, refinement, ablation, output/provenance contract, and four fixed retrieval queries, all stored verbatim and hashed.
- **2.10** D12: exact generation, retrieval-embedding, and matching-embedding versions and hashes where available.

The configuration must reject missing values and write a resolved, read-only snapshot
into every campaign directory.

### Exit gate

- All blocking decisions except final `k_doc` are resolved.
- Changing a frozen artefact changes campaign identity and prevents continuation of
  an existing campaign.

## Phase 3 — Build and validate the research inputs

### Work

- **3.1** Build D1 manifests with `doc_id`, BibTeX key, DOI, title, venue, year, and obtained status. Only `rep_01`…`rep_54` and `fed_01`…`fed_23` go downstream.
- **3.2** Build D2 corpora using opaque filenames keyed by `doc_id`.
- **3.3** Re-encode and freeze both D3 reference models against the new metamodel.
- **3.4** Correct Federation variability before runs: `Arity`, `Exec_mode`, `Reified`, and `Trigger` are alternate groups. Record this pre-campaign data correction.
- **3.5** Build D4 attribution maps: Federation from the published matrix; Repair from Macedo section 3 citations, labelled as reconstructed evidence.
- **3.6** Build D21 feature partitions into `attested` and `organising`.
- **3.7** Generate D22 with the primary ordering for each corpus and two additional seeded Repair orderings. Store seeds and realised lists.
- **3.8** Validate referential integrity among D1, D2, D3, D4, D21, and D22.
- **3.9** Compute ρ(T,C) before generation begins.

### Exit gate

- Counts are exactly 54 Repair and 23 Federation documents, or a documented design
  amendment explains the difference.
- D4 covers all attested features; D21 covers every reference feature; every D4/D22
  `doc_id` exists in D1 and D2.
- No title, author, venue, or ground-truth label leaks through corpus filenames or
  generated prompt paths.

## Phase 4 — Rebuild preprocessing and retrieval

### Work

- **4.1** Extract and normalise text.
- **4.2** Strip references, acknowledgements, biographies, copyright blocks, and other front/back matter before chunking.
- **4.3** Split at sections and paragraphs; split oversized paragraphs at sentences; apply the frozen overlap policy.
- **4.4** Write one deterministic D11 `chunks.jsonl` per corpus with `chunk_id`, `doc_id`, offsets, text, and preprocessing version.
- **4.5** Prefix indexed text with `search_document:` without changing canonical text.
- **4.6** Build exactly two Chroma collections with `id = chunk_id` and `doc_id` metadata.
- **4.7** Implement four fixed `search_query:` sub-queries. Retrieve within the current batch, merge, deduplicate by `chunk_id`, and preserve scores.
- **4.8** Report min, median, mean, and max chunks per document by corpus.
- **4.9** Choose `k_doc` strictly below the relevant chunk count and freeze it. Use `k_step = k_doc × |B_j|`; define how remainders are distributed across four queries.

### Retrieval validity check

- **4.V** Before outcome scores exist, inspect the top ten chunks for three preselected documents. Record whether they represent methods rather than references, related work, or background. If this fails, repair preprocessing/chunking and rebuild D11 and both indexes; do not tune queries against scores.

### Exit gate

- Identical input produces identical chunk IDs and content hashes.
- RAG and Non-RAG resolve their text from the same D11 records.
- Batch filters cannot return chunks from another batch.
- Retrieval inspection passes, `k_doc` is frozen, and index metadata matches D11/D12.

## Phase 5 — Implement the unified granularity engine

### Required behaviour

For a corpus of `n` documents and configured `N`:

- **5.1** Read frozen ordering `π` and partition it into exactly `N` non-empty contiguous slices. Define and test the ceiling-sized policy for non-divisible cases.
- **5.2** At step `j`, assemble the complete previous model (empty only at step 1) and the current batch context.
- **5.3** Non-RAG includes every D11 chunk for the batch in deterministic order, with no selection and no `k`.
- **5.4** RAG includes only batch-scoped merged/deduplicated results using frozen `k_doc`.
- **5.5** Put invariant prompt blocks before variable blocks for prefix caching.
- **5.6** Count tokens before generation. Over-limit requests emit an infeasible record and make no model call.
- **5.7** Reject unresolved placeholders; record truncation when finish reason is `length`.
- **5.8** Persist the model and logs atomically before advancing.

There is no special SS or IS path: `N=1` naturally has no previous model and `N=n` naturally has one document per step.

### Tests

- **5.T1** Each document occurs once and only once for every configured `N`.
- **5.T2** Ordering is reproducible and batching identical across grounding conditions.
- **5.T3** The complete previous model is carried forward.
- **5.T4** Non-RAG includes all and only batch chunks.
- **5.T5** RAG respects filters, prefixes, `k_step`, deduplication, and score logging.
- **5.T6** Context-limit failure does not invoke the model.
- **5.T7** Generation cannot import evaluation or ground-truth modules.
- **5.T8** Resume starts only after a committed step and never overwrites a run made with a different frozen configuration.

### Exit gate

- Tests pass for Repair `N={1,5,10,20,54}` and Federation `N={1,5,10,23}` using a
  fake model and miniature fixtures.
- Phase 1 legacy modules and entry points are removed.

## Phase 6 — Enforce the logging contract

Every attempt uses a stable `run_id` derived from campaign and full configuration. Mandatory artefacts are:

- **6.1** D7 `fm_gen.xml`: final generated model.
- **6.2** D8 `fm_iter/<step_idx>.xml`: every completed-step model, with enough information to derive `first_seen_step`.
- **6.3** D9 `context_log.jsonl`: run, step, `N`, condition, batch IDs, chunk IDs, retrieval scores, query allocation, and `k_step`.
- **6.4** D10 `run_meta.json`: model/version, hashes, temperature/reasoning, seed, ordering, `N`, assembled tokens, tokens in/out, wall time, attempts, finish reason, feasible and truncation flags, and failure details.
- **6.5** D11 hash and D22 ordering identity used by the run.
- **6.6** A validator marks a run complete only when required files are present, parseable, internally consistent, and match frozen campaign hashes.

### Exit gate

- A throwaway run in each grounding condition passes validation.
- Forced infeasibility, truncation, provider error, retry, interruption, and resume
  scenarios have tests.
- Throwaway outputs are deleted before the measured campaign directory is created.

## Phase 7 — Pilot and campaign readiness

- **7.1** Run one open-weight Repair arm at `N=54` to verify the 16,384 output-token cap. This is a systems pilot, not an outcome-scoring exercise.
- **7.2** Confirm actual context windows and token counting for every exact model version.
- **7.3** Estimate call count, token use, cost, and elapsed time from logs.
- **7.4** Confirm rate-limit/backoff behaviour and provider availability.
- **7.5** Generate a manifest containing every planned run before executing any of them. It is the denominator for completion reporting.

If a blocking setting changes, discard affected pilots/indexes, change campaign identity, and repeat the affected gates.

### Exit gate

- No unresolved blocking decision remains.
- The manifest matches the approved matrix and call accounting.
- The measured directory contains only frozen inputs and its manifest.

## Phase 8 — Execute the campaign

Run in this order:

- **8.1** Guided headline arms at `N=10`, both corpora and conditions, all three models, 20 repetitions.
- **8.2** Guided `N=1` baselines, both corpora and conditions, 20 repetitions.
- **8.3** Guided curves: Repair `N={5,20,54}` and Federation `N={5,23}`, both conditions, five repetitions.
- **8.4** Metamodel ablation: RAG only, `N={1,10}`, both corpora, 20 repetitions.
- **8.5** Repair order sensitivity: RAG, `N={10,54}`, two alternate orderings, five reps.
- **8.6** Repair retrieval-depth sweep: RAG, `N=10`, `k_doc={3,5,10,15}`, three reps, subject to the Phase 4 ceiling.

Validate and back up outputs after each block. Do not calculate outcome scores while generation runs; operational monitoring may use completion, failure, tokens, latency, and cost only.

If cuts are necessary, cut the `k` sweep, then order sensitivity, then Federation interior points. Do not cut D8/D9, calibration, criterion divergence, or Repair `N=54`.

### Exit gate

- Every manifest row is `complete`, `infeasible`, or `failed` with a reason.
- There are no orphan steps, hash mismatches, silent truncations, or pilots mixed with
  measured runs.

## Phase 9 — Derive metrics without model calls

All four streams below can run in parallel once Phase 8 outputs are validated. Every stream writes into `results/<campaign_id>/analysis/`, which is separate from the immutable run outputs; analysis can read but cannot mutate run outputs.

Global rules that apply across all streams:

- Semantic scores are computed against **reach** as primary, full F_t as a labelled bound, and attested/organising partitions separately.
- **Parent-match** is the primary structural measure; coverage is reported beside its signals, never in place of them.
- Mann–Whitney U with **Holm** correction over the declared family; **Cliff's δ** alongside adjusted `p`; means with 95% CI; SD per configuration.
- Never rerun generation for any analysis step.

### Stream 9A — Calibration + separated criteria + variability (RQ2, RQ1)

- **9A.1** Regenerate ρ(T,C) (from P3.9) into the campaign snapshot for traceability.
- **9A.2** Dual recall: primary against reach; F_t as labelled bound; F_t^att and F_t^org separately.
- **9A.3** D13 conformance CSV: conformance bool, FeatureIDE parse bool, W1–W5 violations by name.
- **9A.4** D14 SAT CSV: satisfiability, dead features, constraint counts by type.
- **9A.5** D15 match CSV: gen ↔ ref pairs with raw similarity scores, not booleans (τ sweep re-thresholds this file).
- **9A.6** D17 shape CSV: feature count, depth, branching, **group-kind proportions, mandatory ratio, degenerate flag**.
- **9A.7** Variability report: reference-free measures on both corpora; reference-based (group-kind confusion matrix, mandatory agreement) on Repair primary and on Federation once P3.4 encoding fix is in. **Degenerate share is the headline number**.
- **9A.8** **Criterion-divergence rank table** — load-bearing. Rankings under conformance, SAT-family, and semantic criteria must differ; if they coincide, the separation claim is decorative.

### Stream 9B — Granularity + feasibility (H1a, H1b, H2, H4)

- **9B.1** Feasibility boundary report: assembled-token vs window, per model × corpus × N × grounding, from D10.
- **9B.2** Headline H1b test at `N=10`: RAG vs Non-RAG on both corpora, MW U + Holm + Cliff's δ. A null at `N ∈ {10, 20}` is the paper's most quotable outcome.
- **9B.3** N-curve plot per model per grounding, both corpora; identify `N*` saturation and whether it scales proportionally with `n`.
- **9B.4** Model-growth curve (features per step) from D8; check for bloat as failure mode.
- **9B.5** Per-document recall at `N=n` only, using D4 + D9.
- **9B.6** Order-sensitivity variance across three orderings on Repair.

### Stream 9C — Ablation + provenance (H3, RQ6)

- **9C.1** Δconformance under ablation (guided vs no-metamodel) — expect sharp drop.
- **9C.2** ΔF1, Δparent-match, Δdegenerate-share under ablation — expect clean null on semantics.
- **9C.3** Schema token overhead from D10 (pair with authoring-effort note; R1 asked).
- **9C.4** D16 provenance CSV: markers, parse status, cited docs, `first_seen_step`.
- **9C.5** Provenance L0 emission compliance and L1 referential integrity per configuration.
- **9C.6** Provenance L2 hallucination: was the cited `doc_id` in context at `first_seen_step`? Vacuous for Non-RAG at `N=1`; report as a property, not a gap.
- **9C.7** **Recency bias** — critical check. Citation distribution vs batch position. Strong skew invalidates L0–L2.

### Stream 9D — Threshold sensitivity and statistical hygiene

- **9D.1** Rescore D15 at τ ∈ {0.3, 0.4, 0.5, 0.6}. Report whether **rankings** and H1b/H2 verdicts flip, not whether levels move.
- **9D.2** Apply Holm correction across the whole declared family of tests.
- **9D.3** List every previously borderline result that did not survive correction, and report it as not surviving.
- **9D.4** Report confidence intervals and dispersion per configuration alongside every headline number.

### Exit gate

- Every published table is reproducible by one documented analysis command.
- Analysis can read but cannot mutate run outputs.
- Missing, infeasible, failed, and truncated runs are explicitly reported.

## Phase 10 — Human calibration and final release

- **10.1** Freeze rating sample and rubric before ratings begin.
- **10.2** Use two raters and report quadratic weighted Cohen's κ.
- **10.3** Compare expert and Top-FM ordering with rank correlation; report divergence as evidence about α/β, not as a defect.
- **10.4** Produce criterion-divergence and hypothesis-decision tables, including null and non-significant results after correction.
- **10.5** Record corpus, context-window, reconstructed-attribution, model-version, and campaign-deviation limitations.
- **10.6** Final hygiene: no old 2×2 output, obsolete entry point, invalid index, pilot, secret, or cache in the release.
- **10.7** Publish checksums, resolved config, environment lock, manifest, data dictionary, and exact reproduction commands with allowed artefacts.

### Final definition of done

- The repository exposes one preprocessing path, one generation engine, one campaign
  runner, and one analysis workflow.
- Required D1–D22 artefacts exist or are explicitly not applicable; no required
  run-time log was reconstructed after the fact.
- The active tree has no superseded results, chunks/indexes, duplicate canonical
  inputs, or unnecessary SS/IS modules, scripts, tests, and documentation.
- A clean checkout reproduces preprocessing and analysis without untracked local state.

## Phase gate summary

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| 0 | Recoverable baseline and inventory | Deletions classified |
| 1 | Clean derived state and lock code-removal list | Old data/results gone; migration targets recorded |
| 2 | Frozen protocol | Blocking choices fixed except measured `k_doc` |
| 3 | D1–D4, D21, D22 and rho(T,C) | Cross-artefact integrity passes |
| 4 | D11 and two valid indexes | Retrieval inspection and `k_doc` freeze pass |
| 5 | Unified `N`-step engine | Behaviour tests pass; legacy runners removed |
| 6 | D7–D10 logging | Throwaway validation passes |
| 7 | Pilot and run manifest | Campaign readiness approved |
| 8 | Immutable measured runs | Every planned run accounted for |
| 9 | D13–D19 via parallel streams 9A/9B/9C/9D | Tables reproducible; run outputs untouched |
| 10 | Human calibration and release | Hygiene and reproducibility pass |
