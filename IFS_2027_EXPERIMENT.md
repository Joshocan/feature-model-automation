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
  experiment.yamlw
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
- **6.6** The validator reports execution completion, XSD validity, and overall
  admissibility separately. It also checks required files, cross-artefact
  consistency, and frozen campaign hashes.

### Exit gate

- A throwaway run in each grounding condition passes validation.
- Forced infeasibility, truncation, provider error, retry, interruption, and resume
  scenarios have tests.
- Throwaway outputs are deleted before the measured campaign directory is created.

## Phase 7 — Pilot and campaign readiness

- **7.1** The systems pilots established a 32,768-token output allowance before
  the measured campaign. Truncation remains a terminal outcome and is never
  carried forward.
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

The two open-weight models (GLM 5.3 Flash and DeepSeek V4.1 Flash) run the
full matrix below. GPT-6 Astra is a reduced proprietary ceiling: Repair
`N=10` RAG and Non-RAG (10 reps), `N=10` RAG ablation on both corpora
(10 reps), and Federation `N=10` guided RAG (10 reps). Its existing Repair
`N=5` RAG five-seed characterization and `N=1` Non-RAG capability probe are
retained separately and are not relabelled as new campaign rows.

Run the open-weight lane in this order:

- **8.1** Guided headline arms at `N=10`, both corpora and conditions, both open-weight models, 20 repetitions.
- **8.2** Guided `N=1` baselines, both corpora and conditions, 20 repetitions.
- **8.3** Guided curves: Repair `N={5,20,54}` and Federation `N={5,23}`, both conditions, five repetitions.
- **8.4** Metamodel ablation: RAG only, `N={1,10}`, both corpora, 20 repetitions.
- **8.5** Repair order sensitivity: RAG, `N={10,54}`, two alternate orderings, five reps.
- **8.6** Repair retrieval-depth sweep: RAG, `N=10`, `k_doc={3,5,10,15}`, three reps, subject to the Phase 4 ceiling.

The open-weight lane contains 644 runs / 6,300 calls. The new Astra lane
contains 50 runs / 500 calls. They may run concurrently because they use
different providers; GLM and DeepSeek remain sequential within the shared
Ollama lane. The combined new denominator is 694 runs / 6,800 calls.

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

- **10.1** Use the versioned [expert-evaluation strategy](docs/expert-evaluation-strategy.md) and freeze its rubric, sample rule, and blinded code key before expert ratings. The added Astra `N=1` RAG outputs are a labelled expert-study extension, not part of the original campaign matrix.
- **10.2** Seek three independent domain-qualified raters. Report agreement for the four ordinal quality criteria and the blinded triplet preferences, plus the two within-rater repeats descriptively.
- **10.3** Compare independent expert quality judgments and matched three-model first-draft preferences with the automatic criteria after the ratings are locked. Do not describe this sample as a Top-FM-versus-median experiment or infer model-weight causality from provider-specific settings.
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

---

## Runnable commands (terminal cheat sheet)

Every command assumes:

```bash
cd /Users/joshuaocansey/dev/feature-model-automation
source $HOME/.venvs/fame/bin/activate
```

All scripts read paths and settings from `config/experiment.yaml`. Any change
to a hashed artefact (prompts, metamodel, chunks, encoders) invalidates the
frozen protocol — regenerate `data/frozen/protocol.sha256` afterwards.

### Phase 3 — Build and freeze research inputs

D1/D3 already committed. To (re)generate D4, D21, D22, ρ from the current
manifests + ground-truth XMLs + attribution seed:

```bash
python scripts/phase3_build_inputs.py    # if not already committed
# OR the individual helpers if kept separately.
```

**Verify integrity:**

```bash
python -c "
import json
r = json.load(open('data/calibration/rho.json'))
for c in ('federation','repair'):
    print(c, r[c])
"
```

Expected: federation `ρ ≈ 0.28`, repair `ρ ≈ 0.58`.

### Phase 4a — Build canonical chunk store (D11)

Runs `unstructured` on every PDF in `data/raw/{corpus}/` and writes the frozen
`chunks.jsonl` per corpus.

```bash
# Both corpora
python scripts/build_chunks.py

# One corpus only
python scripts/build_chunks.py --corpus federation
python scripts/build_chunks.py --corpus repair --limit 3   # debug: first 3 docs
```

**What to watch:**

* Per-PDF one line: `[fed_XX] N elems | XXk chars kept | K chunks | Ts`
* Final stats table + writes `data/processed/chunks_stats.json`
* Federation should produce ~23 docs / ~1,000 chunks; Repair ~54 / ~2,700

**Verify:**

```bash
wc -l data/processed/*/chunks.jsonl
cat data/processed/chunks_stats.json | python -m json.tool | head -20
```

### Phase 4b — Build Chroma index + retrieval validity check

Prerequisite: Ollama running with `nomic-embed-text` pulled locally.

```bash
# Index both corpora into data/chroma/{federation,repair}/
python scripts/build_index.py

# One corpus only
python scripts/build_index.py --corpus federation
```

Expected wall time: ~2 min federation, ~6 min repair. Each chunk is embedded
serially via Ollama's `/api/embeddings`. Zero failures expected.

**Retrieval validity (4.V smoke, no LLM required):**

```bash
python scripts/validate_retrieval.py
# writes results/smoke/retrieval_validity.md
```

Inspect the top-10 chunks per preselected doc: they should be methodology,
not references / related work.

### Phase 5 — Unit tests (all engine behaviour)

```bash
python -m pytest tests/ -q
```

Expected: **80 pass, 1 skipped, 0 failed**.

### Phase 6 — Smoke run + validator

Full end-to-end with FakeLLM (deterministic, no network cost):

```bash
python scripts/smoke_run.py \
  --campaign-id smoke-$(date +%Y-%m-%d) \
  --corpus federation --N 1 --grounding nonrag \
  --provider fake \
  --validate --force
```

**Live LLM smoke (uses Ollama Pro credits):**

```bash
python scripts/smoke_run.py \
  --campaign-id smoke-minimax-m3 \
  --corpus repair --N 3 --grounding rag \
  --provider ollama_cloud --model minimax-m3:cloud \
  --host https://ollama.com \
  --max-output-tokens 32768 \
  --validate --force
```

**GPT-OSS 120B candidate pilot (not included in the campaign matrix):**

```bash
.venv/bin/python scripts/smoke_run.py \
  --campaign-id pilot-2026-09-22-gpt-oss-120b-n5 \
  --corpus repair --N 5 --grounding rag \
  --provider ollama_cloud --model gpt-oss:120b-cloud \
  --host https://ollama.com \
  --max-output-tokens 32768 \
  --validate --force
```

**GLM 5.3 Flash candidate retry with the improved prompt:**

```bash
.venv/bin/python scripts/smoke_run.py \
  --campaign-id pilot-2026-09-22-glm-5-3-flash-n5-prompt-v2 \
  --corpus repair --N 5 --grounding rag \
  --provider ollama_cloud --model glm-5.3-flash:cloud \
  --host https://ollama.com \
  --max-output-tokens 32768 \
  --validate --force
```

**DeepSeek V4 Pro candidate pilot:**

```bash
.venv/bin/python scripts/smoke_run.py \
  --campaign-id pilot-2026-09-22-deepseek-v4-pro-n5 \
  --corpus repair --N 5 --grounding rag \
  --provider ollama_cloud --model deepseek-v4-pro:cloud \
  --host https://ollama.com \
  --max-output-tokens 32768 \
  --validate --force
```

**Validate a specific run directory:**

```bash
python scripts/validate_run.py \
  results/smoke-*/federation/*/*/
```

Exit code 0 = complete + consistent + hashes match. Non-zero = at least one
ERROR finding.

### Phase 7 — Probe models, build matrix, estimate cost

**7.2 + 7.4 — probe all campaign models:**

```bash
# Basic reachability + tokenizer alignment (cheap: ~$0.01 total)
python scripts/probe_models.py --no-burst

# With rate-limit / backoff probe (5 quick-fire calls per model)
python scripts/probe_models.py --burst 5

# Only one model
python scripts/probe_models.py --only minimax_m3
python scripts/probe_models.py --only gpt_oss_120b
python scripts/probe_models.py --only glm_5_3_flash
python scripts/probe_models.py --only deepseek_v4_pro
```

Report saved to `results/pilot-<date>/model_probe.json`. Exits non-zero if
any configured model is unreachable.

**7.5 — build the run manifest:**

```bash
# Combined frozen manifest
./.venv/bin/python scripts/build_run_matrix.py --only-enabled --lane all

# Disjoint provider-lane manifests
./.venv/bin/python scripts/build_run_matrix.py --only-enabled --lane open_weight
./.venv/bin/python scripts/build_run_matrix.py --only-enabled --lane astra

# Preview without writing to disk
./.venv/bin/python scripts/build_run_matrix.py --only-enabled --lane all --dry-run
```

The accepted totals are 694/6,800 combined, 644/6,300 open-weight, and
50/500 Astra. Matrix generation fails when those totals drift.

**7.1 — systems pilot:** completed before freeze. The resulting open-weight
allowance is 32,768 tokens; do not rerun pilots into the measured directory.

**7.3 — extrapolate cost from pilot data:**

```bash
python scripts/estimate_cost.py \
  --matrix results/ifs-2027/run_matrix.json \
  --pilot  results/pilot-2026-09-22/repair/*/*/
```

Prints per-model + per-arm cost + wall projections.

### Phase 8 — Interactive campaign runner

The main event. Menu-driven, per-step verbose, per-arm validated,
resume-safe.

**Regenerate and verify the two lane manifests first:**

```bash
./.venv/bin/python scripts/build_run_matrix.py --only-enabled --lane open_weight
./.venv/bin/python scripts/build_run_matrix.py --only-enabled --lane astra
```

**Run these in two terminals for provider-level parallelism:**

```bash
# Terminal 1: GLM and DeepSeek sequentially through Ollama
./.venv/bin/python scripts/campaign.py \
  --matrix results/ifs-2027/run_matrix_enabled_open_weight.json \
  --lane open_weight

# Terminal 2: Astra through OpenAI
./.venv/bin/python scripts/campaign.py \
  --matrix results/ifs-2027/run_matrix_enabled_astra.json \
  --lane astra
```

The menu shows each arm with remaining-calls, done/total, est. cost, est.
wall time. Select `1`–`9` for one arm, `A` for all arms in order, `Q` to
quit. Every arm asks `Proceed? [y/N]` before firing. Per-step live output.

**Run one arm only (still confirms before spending):**

```bash
./.venv/bin/python scripts/campaign.py \
  --matrix results/ifs-2027/run_matrix_enabled_open_weight.json \
  --lane open_weight \
  --arm guided_baseline
```

Open-weight arm names: `guided_baseline`, `guided_headline`, `guided_curve`,
`ablation`, `order_sensitivity`, `k_doc_sweep_k3`, `k_doc_sweep_k5`,
`k_doc_sweep_k10`, `k_doc_sweep_k15`. Astra uses `guided_headline`,
`ablation`, and `astra_cross_corpus`.

**Non-interactive (batch mode; opt-in with `--yes`, use with care):**

```bash
./.venv/bin/python scripts/campaign.py \
  --matrix results/ifs-2027/run_matrix_enabled_open_weight.json \
  --lane open_weight \
  --run-all --yes
```

**Skip per-run validator** (validate later with `scripts/validate_run.py`):

```bash
python scripts/campaign.py --matrix ... --arm ablation --no-validate
```

**Force overwrite** of existing run outputs (default is resume-skip):

```bash
python scripts/campaign.py --matrix ... --arm guided_headline --force-runs
```

**Resume after interruption:** re-run the same command. Every persisted
terminal outcome—including malformed, truncated, infeasible, and provider
failure—is retained in the denominator. A process-interrupted, non-terminal
run requires an explicit `--force-runs` restart. Advisory locks prevent two
local processes from owning the same row concurrently.

**Watch progress in a second terminal:**

```bash
watch -n 30 '
    echo "== runs completed =="; \
    find results/ifs-2027 -name "fm_gen.xml" | wc -l; \
    echo "== last run =="; \
    ls -tr results/ifs-2027/*/*/*/fm_gen.xml 2>/dev/null | tail -1
'
```

### Phase 9 — Derived metrics (after Phase 8 completes)

Phase 9 reads from `results/<campaign_id>/**/*` and writes into
`results/<campaign_id>/analysis/` — the raw run outputs are never mutated.
Each driver is documented in its own guide; the commands below are the
end-to-end pipeline.

```bash
# Phase 1 — inventory
python scripts/inventory_campaign.py --output results/ifs-2027/analysis/inventory-<date>

# Phase 2 — structural + logical
python scripts/evaluate_structure.py --inventory results/ifs-2027/analysis/inventory-<date> \
                                      --output    results/ifs-2027/analysis/structure-<date>

# Phase 3 — semantic (requires D03 encoder identity)
python scripts/evaluate_semantic.py  --inventory results/ifs-2027/analysis/inventory-<date> \
                                      --output    results/ifs-2027/analysis/semantic-<date>

# Phase 4 — provenance (L0/L1/L2 + recency)
python scripts/evaluate_provenance.py --inventory results/ifs-2027/analysis/inventory-<date> \
                                       --output    results/ifs-2027/analysis/provenance-<date>

# Phase 5 — join + variability + criterion divergence
python scripts/aggregate_campaign.py --structural results/ifs-2027/analysis/structure-<date> \
                                      --semantic   results/ifs-2027/analysis/semantic-<date> \
                                      --provenance results/ifs-2027/analysis/provenance-<date> \
                                      --output     results/ifs-2027/analysis/campaign-<date>

# Phase 6 — statistical hygiene, τ sweep, plotting
python scripts/analyse_family.py --wide results/ifs-2027/analysis/campaign-<date>/wide.csv \
                                  --family config/analysis/families/<name>.json \
                                  --output results/ifs-2027/analysis/family-<name>-<date>
python scripts/tau_rescore.py    --pairs results/ifs-2027/analysis/semantic-<date>/pairs.csv \
                                  --output results/ifs-2027/analysis/tau-sweep-<date>
python scripts/plot_phase6.py    --campaign results/ifs-2027/analysis/campaign-<date> \
                                  --output   results/ifs-2027/analysis/plots-<date>
```

### Common recipes

**Regenerate protocol hashes after changing any frozen artefact:**

```bash
shasum -a 256 \
  prompts/fm_prompt_template.txt \
  prompts/feature-model-schema.xsd \
  config/experiment.yaml \
  data/encoder_versions.txt \
  fame/utils/marker_grammar.py \
  data/raw/federation/manifest_fed.csv \
  data/raw/repair/manifest_repair.csv \
  data/ground_truth/federation.xml \
  data/ground_truth/repair.xml \
  data/attribution/federation.csv \
  data/attribution/repair.csv \
  data/feature_partition/federation.csv \
  data/feature_partition/repair.csv \
  data/orderings.json \
  data/calibration/rho.json \
  data/processed/federation/chunks.jsonl \
  data/processed/repair/chunks.jsonl \
  data/processed/chunks_stats.json \
  > data/frozen/protocol.sha256
```

**Show what the full test suite currently covers:**

```bash
python -m pytest tests/ --collect-only -q
```

**Kill a stuck run cleanly:**

```bash
pkill -f "smoke_run.py\|campaign.py"
```

Any run that was mid-step will be resume-skipped on next attempt because
`fm_gen.xml` won't exist for it — the resume guard only skips complete runs.
