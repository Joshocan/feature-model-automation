# Implementation Brief: What to Run, Log, Compute and Expect

Supersedes `evaluation-plan-table.md`. Written against the §3–§5 drafts plus the design changes agreed since.

---

## 0. What changed, and why

**The 2×2 is gone, replaced by one continuous axis.** Single-stage and document-incremental are not two strategies. They are the endpoints of refinement granularity $N$, the number of generation steps. $N = 1$ is single-stage. $N = n$ is document-incremental. Everything between is batched refinement, which the old code never ran.

This removes a confound the old design could not address: moving from Federation to Repair changed document count and step count together, so H4 was unanswerable. With $N$ swept, evidence volume and refinement count vary independently.

It also answers the operational objection. Nobody runs 1000 steps on 1000 documents. The finding the study can now produce is $N^{*}$, the granularity past which refinement stops paying.

Six further corrections to the old plan:

1. **The Federation reference model is third-party**, Amrani et al. MODELS Companion 2024, with a published per-study classification matrix. The old plan called it in-house, which would have undercut the ground-truth-independence argument.
2. **Reference features are partitioned** into attested (`F_t^att`, carrying per-study cells) and organising (`F_t^org`, the survey authors' framing). The attribution map is total on the attested part only. Recall is reported over both separately. New artefact D21.
3. **Attribution has two sources.** Federation: read directly off the published matrix. Repair: reconstructed from inline citation lists in Macedo §3, which is weaker and must be labelled as such.
4. **Coverage is demoted.** Parent-match is the primary structural measure. `w_f`/`w_p` therefore need no sweep.
5. **Chunking is rebuilt.** 6000-character chunks gave diffuse embeddings. New size is 1,200–1,500 characters at 10–15% overlap, references and front and back matter stripped, split on paragraph boundaries; retrieval uses four fixed sub-queries with the mandatory `nomic` prefixes. Largest available gain in retrieval quality, and it invalidates any existing index.
6. **τ gets a sweep and it is free**, being a rescore of stored similarity scores. Never select τ on outcome.

---

## 1. Design, precisely

### Granularity
Corpus of $n$ documents, fixed ordering $\pi$. At granularity $N$, the corpus is partitioned into $N$ **contiguous slices** of $\pi$ of size $\lceil n/N \rceil$. Every document is read exactly once at every $N$. Batch membership is never randomised; $\pi$ is the only source of order variation and it is held fixed except in the order-sensitivity check.

| Corpus | $N$ values | Docs per step |
|---|---|---|
| Repair ($n=54$) | 1, 5, 10, 20, 54 | 54, 11, 6, 3, 1 |
| Federation ($n=23$) | 1, 5, 10, 23 | 23, 5, 2, 1 |

$N = 20$ on Repair is not optional. Without a point between 10 and 54, a plateau beginning at 10 and one beginning at 30 look identical.

### The loop
At step $j$: context is batch $B_j$ plus $FM^{(j-1)}$ serialised in full. Identical at every $N$. At $N = 1$ there is no previous model, which is what makes single-stage the degenerate end rather than a separate mechanism.

### Grounding — the only thing that differs inside a step
Both conditions batch identically. **Non-RAG** takes every chunk of $B_j$, in document order, no selection, no $k$. **RAG** takes the top-$k$ chunks of $B_j$.

### Chunking — the largest single lever on retrieval quality

**6000 characters is too long, and it is inherited rather than chosen.** That is roughly 1,500 tokens, spanning several topics, so its embedding is an average of all of them: it matches many queries weakly and none strongly. Dense retrieval works best at 200 to 800 tokens, where one vector represents one idea.

Drop to roughly **1,200 to 1,500 characters with 10 to 15 percent overlap**. Total corpus text is unchanged, so Non-RAG is unaffected except that it sees more, smaller pieces. Two knock-ons: chunks-per-document rises, which raises the ceiling on $k_{\mathrm{doc}}$, and $k_{\mathrm{doc}}$ must be rescaled so a step still receives a comparable amount of text.

### Preprocessing hygiene

Strip references, acknowledgements, author biographies and copyright blocks **before** chunking. A references section is a dense bag of domain vocabulary that retrieves well and contributes nothing, and worse, it gets cited in traces, inflating provenance with attributions that are true and meaningless.

Split on paragraph and section boundaries rather than fixed offsets, so no chunk begins mid-sentence.

### Retrieval

Queries are **fixed**: identical at every step, in both RAG arms, independent of generation state. Never query with the current feature model; that confounds H1b with a feedback effect.

Use **four fixed sub-queries** rather than one long string. Each is sharper than a single vague query, and none names taxonomy vocabulary, so nothing leaks beyond `{{DOMAIN}}`, which the prompt already discloses:

```python
QUERIES = [
  "search_query: what {domain} approach this paper proposes and what it operates on",
  "search_query: how the approach works, its mechanism, algorithm or procedure",
  "search_query: the options, modes, variants and alternatives the approach supports",
  "search_query: what the approach requires, assumes, or guarantees",
]
```

Retrieve $k_{\mathrm{step}}/4$ per sub-query against the same `doc_id` filter, then merge and deduplicate by `chunk_id`.

**Prefixes are mandatory.** `nomic-embed-text` is trained with task prefixes and degrades silently without them: `search_query:` on queries, `search_document:` on chunks. If the index was built without prefixes, rebuild it.

Set $k_{\mathrm{step}} = k_{\mathrm{doc}} \times |B_j|$ so each document contributes about the same number of chunks at every $N$.

**Check before running:** $k_{\mathrm{doc}}$ must be below the typical chunks-per-document count, or at $N = n$ retrieval returns everything and RAG becomes Non-RAG. The old per-source grid reached $k = 25$, which at 6000-character chunks was likely more chunks than one paper yields, so those arms may have been the same pipeline. Measure chunks-per-document per corpus and report it.

**Validity check, before any scores exist.** Take three documents, run retrieval, read the top ten chunks. They should come from methodology and approach sections, not related work and background. This is a judgement about content made before any number exists, so it cannot contaminate the comparison. If the top chunks are mostly related work, the chunks are too large or the references were not stripped. Neither is fixed by rewriting the query.

**Do not tune the queries against results.** Retrieval sits inside the pipeline under test, so selecting the query that maximises F1 is the same error as selecting $\tau$ that way, and worse. Fix the strings before the campaign and print them verbatim in the paper.

### Storage
One canonical chunk store per corpus; Chroma is a derived index over it.

```
corpus/ (D2) → chunks.jsonl (D11) → chroma collection
               chunk_id, doc_id,     id = chunk_id
               offsets, text         embedding + {doc_id}
```

Two collections total, not one per $N$. Batching is a metadata filter at query time:

```python
hits = {}
for q in QUERIES:                                  # four fixed sub-queries
    r = collection.query(query_texts=[q],
                         n_results=k_step // len(QUERIES),
                         where={"doc_id": {"$in": batch_doc_ids}},
                         include=["documents", "metadatas", "distances"])
    for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0],
                                    r["metadatas"][0], r["distances"][0]):
        hits.setdefault(cid, (doc, meta, dist))     # dedup by chunk_id
```

At $N=1$ the filter covers everything; at $N=n$ one document. No special-casing in the loop.

Both arms resolve chunk text from the same JSONL. This is not tidiness: if the two arms chunk separately, any RAG advantage could be a chunking artefact and H1 becomes untestable.

### Model selection and its consequence for H1a

All three selected models carry 1M context windows. Repair at $N = 1$ under Non-RAG is roughly 500k tokens, so **it now fits**, and the capacity boundary H1a was built around does not bite at these corpus sizes.

This improves the study rather than damaging it. H1b becomes cleanly testable at every granularity including $N = 1$, with no capacity confound anywhere, so the retrieval comparison is purely about whether selection beats supplying everything. Reframe H1a as a measured statement: report assembled token counts and state that capacity was not binding at 54 documents, which bounds the corpus size over which the conclusions hold.

The two open models are matched to each other (42 and 40) so that a claim holding across both means something; the proprietary model is deliberately unmatched at 51, because it is a ceiling rather than a sample.

### Infeasibility
Non-RAG at low $N$ will exceed context windows. Decide now: if the assembled context does not fit, record the run **infeasible**. Do not truncate silently. Log assembled token count in D10 for every run so the boundary is measured rather than inferred.

---

## 2. Hypotheses

> **H1a (Capacity).** Prompt-based conditioning becomes infeasible below some granularity as corpus size grows; retrieval remains feasible throughout. The failure point depends on the model's context window.
>
> **H1b (Selection).** Where both are feasible, retrieval still improves semantic quality, because similarity-based selection raises the proportion of relevant context.
>
> **H2 (Granularity).** Quality improves as $N$ increases from 1.
>
> **H3 (Metamodel guidance).** Removing the metamodel block lowers conformance sharply and leaves semantic precision and recall largely unchanged. A clean null on the semantic criteria is the predicted result.
>
> **H4 (Saturation).** Improvement saturates at some $N^{*}$ well below $n$, and $N^{*}$ does not grow proportionally with corpus size, so linear refinement is not repaid at scale.

H1a is nearly definitional, but the boundary is a practical number nobody reports. H1b is the contestable claim. **The sharpest possible outcome is a null on H1b**: if the RAG advantage vanishes at $N = 10$ and $N = 20$ where both arms fit, retrieval buys capacity and nothing else, and the earlier result was an artefact of testing where Non-RAG was handicapped.

Expect RAG and Non-RAG to converge as $N$ rises, since small batches leave little to select from.

---

## 3. Freeze before any run

Unrecoverable once the campaign starts; every model is conditioned on these.

| ID | Artefact | Note |
|---|---|---|
| **D5** | `metamodel.ecore` + literal prompt text + hash | **This is M4 and it gates everything.** One byte changes and all runs are void. |
| **D6** | `prompts/` — extraction, refinement, ablation variant, **fixed retrieval query**, each hashed | Includes the verbatim preserve-then-refine policy quoted in §4.2 |
| **D1** | `manifest.csv` per corpus | `doc_id`, bibtex key, DOI, title, venue, year, obtained yes/no |
| **D2** | `corpus/` text keyed by `doc_id` | Opaque filenames. A filename with the paper title leaks the answer into any path reaching a prompt. |
| **D3** | `fm_ref.xml` per corpus, frozen serialisation | Both reference models re-encoded against the new metamodel |
| **D4** | `attribution.csv` — `gt_feature_id` → set of `doc_id` | Federation from the matrix; Repair reconstructed from §3 citations |
| **D21** | `feature_partition.csv` — `gt_feature_id` → {attested, organising} | **New.** Federation: organising = columns with no cells. Repair: the five facet branches per §3.1. |
| **D22** | `orderings.json` — $\pi$ per corpus, plus 2 alternates with seeds | **New.** Fix seeds in advance, not at run time. |
| **D12** | `encoder_versions.txt` | Pinned `all-mpnet-base-v2` and `nomic-embed-text` versions and hashes |

**Compute $\rho(T,C)$ as soon as D3, D4 and D21 exist.** No runs needed. It is the input to the Proposition, and if $\rho$ on Repair is very low you want to know before spending the compute.

---

## 4. Run matrix

Decisions taken: ablation on RAG only at the headline granularity; order sensitivity on Repair only.

**Guided campaign** — 3 models, 2 grounding conditions, both counted below.

| Corpus | $N$ | Reps | Role | Calls |
|---|---|---|---|---|
| Repair | 1 | 20 | single-stage baseline | 120 |
| Repair | 10 | 20 | **headline comparison, H1b tested here** | 2,400 |
| Repair | 5, 20, 54 | 5 | curve, H2 and H4 | 2,370 |
| Federation | 1 | 20 | baseline | 120 |
| Federation | 10 | 20 | headline comparison | 2,400 |
| Federation | 5, 23 | 5 | curve | 840 |
| | | | | **8,250** |

**Remaining arms**

| Arm | Scope | Calls |
|---|---|---|
| Ablation (metamodel removed) | RAG only, $N \in \{1, 10\}$, both corpora, 20 reps | 1,320 |
| Order sensitivity | Repair, RAG, $N \in \{10, 54\}$, 2 extra orderings, 5 reps | 1,920 |
| Retrieval depth | Repair, RAG, $N = 10$, 4 values of $k_{\mathrm{doc}}$, 3 reps | 360 |
| | **Campaign total** | **≈ 11,850** |

Non-RAG runs that come out infeasible never execute, but with all three models at 1M context none are expected at these corpus sizes.

Roughly two-thirds of these calls go to the two open-weight models, which Ollama Pro credits cover across two billing cycles. GPT-6 Astra carries about 3,950 calls and is the real spend; it is also the first arm to trim, for instance by running it only at the headline granularity $N = 10$.

$N = 10$ is where the 20-repetition comparison lives, which means H1b is tested at a granularity where both arms are feasible and which someone would actually deploy. Say once in §5 that the headline moved from $N = n$ to $N = 10$, so nobody silently compares against the old IS-RAG numbers.

---

## 5. Logging contract

Written during the run, not reconstructable. **Verify all of these on one throwaway run before launching.**

| ID | Artefact | Why it cannot be recovered |
|---|---|---|
| **D7** | `fm_gen.xml`, final model per `run_id` | — |
| **D8** | `fm_iter/`, model at **every** step, keyed `run_id` + `step_idx` | The saturation curve lives here. H4 collapses without it. |
| **D9** | `context_log.jsonl`: `run_id`, `step_idx`, $N$, config, llm, `batch_doc_ids`, `chunk_ids`, retrieval scores, $k_{\mathrm{step}}$ | All provenance depends on it. Chunk IDs alone are insufficient; `doc_id` resolution is required. |
| **D10** | `run_meta.json`: llm + version, temperature, seed, prompt hash, **assembled token count**, tokens in/out, wall-clock, $\pi$ used, $N$, feasible flag | Token counts feed H1a and the ablation cost argument |
| **D11** | `chunks.jsonl` | Canonical text of record for both arms |
| **D22** | orderings actually used | — |

Add `first_seen_step` per feature to D8 or make it derivable: provenance L2 needs to know which step a feature first appeared at, to check the cited document was in context *then*.

---

## 6. Derived computations

No LLM calls. Recomputable, so get them right rather than fast.

- **D13** `conformance.csv` — conformance bool, FeatureIDE parse bool, violated conditions by name (W1–W5)
- **D14** `sat.csv` — satisfiable bool, dead features, constraint counts by type
- **D15** `match.csv` — generated ↔ reference pairs with **similarity score, not the match boolean**. The τ sweep re-thresholds this file; storing only decisions means recomputing embeddings four times.
- **D16** `provenance.csv` — `run_id`, `feature_id`, `cited_doc_ids`, `parse_status`, `first_seen_step`
- **D17** `shape.csv` — feature count, depth, branching factor, **group-kind counts and proportions (`and`/`or`/`alt`), mandatory ratio, degenerate flag**
- **D18/D19** — expert ratings, error coding

---

## 7. Expected results

### Calibration (§6.1) — do first, needs no runs
| Measure | Expected |
|---|---|
| $\rho(T,C)$ per corpus | Well below 1 on Repair; near 1 on Federation if all 23 held |
| Recall vs reach, vs full $F_t$ | Recall against reach materially higher; the gap quantifies previously unattributed error |
| Recall on $F_t^{att}$ vs $F_t^{org}$ | **Large gap.** Organising features recovered rarely. If the gap is small, the partition argument is wrong. |

### Separated criteria (§6.2)
Conformance high when guided, collapsing under ablation. Parse rate tracks conformance; divergence means schema-valid but tool-invalid, itself reportable. Satisfiability near 1 on Federation (vacuous, report as such), real variation on Repair only. Cross-tree constraint recovery substantially under-generated. Shape drifts oversized and flatter as $N$ rises.

**Variability.** Expect poor group-kind agreement, with the confusion matrix showing collapse toward `and`. Expect a non-trivial degenerate share, higher in the ablation arm: if removing the schema flattens variability, that is H3 appearing somewhere other than conformance. Mandatory ratio is worth watching against the old prompt's default-to-optional rule, which the new neutral wording should shift.

**The load-bearing table is criterion divergence: rankings under the three criteria must differ.** If they coincide, the separation claim is decorative.

### Granularity (§6.3, H2/H4)
| Measure | Expected |
|---|---|
| Score vs $N$ | Rise then saturate at $N^{*} \ll n$ |
| Feature count vs $N$ | Monotonic growth; bloat is the failure mode |
| $N^{*}$ on Repair vs Federation | Does not scale proportionally with $n$ — this is the practical result |
| Per-document recall (only at $N=n$) | Well below 1. Sharpest test of whether refinement integrates evidence |
| Variance across 3 orderings | Non-trivial |

### Feasibility and grounding (§6.4, H1a/H1b)
Report the granularity at which Non-RAG becomes infeasible, per model and corpus. Then compare arms where both are feasible. **Expect convergence at high $N$.** A null on H1b at $N \in \{10, 20\}$ is the most quotable outcome available.

Mann-Whitney U with Holm correction. Expect some previously borderline results not to survive, and report them as not surviving.

### Ablation (§6.5, H3)
Sharp conformance drop, little or no semantic change, small parent-match degradation. Token overhead of the schema block reported alongside authoring effort, since R1 asked about cost.

### Provenance (§6.6)
L0 emission compliance varies substantially by model. L1 referential integrity high but not 1. L2 context availability gives the **provenance hallucination rate**, non-zero throughout, lower under RAG. Vacuous for Non-RAG at $N=1$; report as a property, not a gap. **Recency bias is the critical check**: strong skew toward recently seen batches means attribution is positional and L0–L2 become uninterpretable.

### Threshold sensitivity (§6.7 + Appendix C)
Rescore over τ ∈ {0.3, 0.4, 0.5, 0.6}. Report whether the **ordering** and the H1b/H2 verdicts hold, not whether levels move. Levels always fall as τ rises; that is not a finding. A ranking flip is.

### Human calibration (§6.8)
Two raters, written rubric, quadratic weighted Cohen's κ. Moderate agreement expected, weakest on semantic judgments. Rank correlation between expert and Top-FM ordering: **expect divergence**, and report it as evidence about α and β rather than as a defect.

---

## 8. Rules to hold under pressure

1. **Never select τ, $k$, $N$, α, β or $w$ on outcome.** τ maximised gives τ = 0, where everything matches.
2. **Never let evaluation reach the generation loop.** No conformance result, no reference score, no ground truth in any prompt or filename.
3. **Report the null.** H3 predicts no semantic effect; a null on H1b would be the paper's best sentence.
4. **Report what does not survive correction.**
5. **Log first, run second.** One throwaway run, verify D7–D11 and D22, then launch.

---

## 9. Critical path

```
Freeze D5, D6              →  nothing runs until the metamodel is final
D1, D2, D3, D4, D21, D22   →  parallel with the freeze
ρ(T,C)                     →  compute now; no runs needed
chunks.jsonl → chroma      →  one store per corpus, index derived
Measure chunks-per-doc     →  sets k_doc; do before the k sweep
Smoke run                  →  verify the logging contract
Guided campaign            →  proprietary models first (rate limits)
Ablation, order, k sweeps  →  reduced repetitions
Derived metrics            →  D13–D17, iterate freely
τ sweep                    →  free rescore of D15
Human protocol             →  start the rubric early; raters are the long pole
```

If the schedule slips, cut in this order: the $k$ sweep, then order sensitivity (move to threats), then the Federation curve interior points. **Do not cut** D8, D9, the calibration section, the criterion-divergence table, or $N = 54$ on Repair. Without the expensive endpoint there is no evidence for saturation, and saturation is the contribution.

---

## 10. Complete configuration reference

Grouped by method and instrument, the distinction §5.3 and §5.5 of the paper now rely on. Method parameters are part of what is being evaluated. Instrument parameters are part of what evaluates it. Rows marked **TBD** are not yet decided and are collected again in §11.

### 10.1 Corpus and data

| Parameter | Value |
|---|---|
| Corpora | Model Repair ($n=54$), Model Federation ($n=23$) |
| Document identifiers | `rep_01`…`rep_54`, `fed_01`…`fed_23` |
| Filenames | Opaque; never contain titles, authors or venue |
| Document ordering $\pi$ | One fixed seeded ordering per corpus, recorded in D22 |
| Alternate orderings | 2 further seeded orderings, Repair only |

### 10.2 Preprocessing — shared by both grounding conditions

| Parameter | Value |
|---|---|
| Text extraction | PDF to plain text, normalised whitespace |
| Text cleaning | Strip references, acknowledgements, author biographies, copyright blocks **before** chunking |
| Split boundaries | Paragraph and section boundaries, never fixed offsets |
| Maximum chunk size | **1,200–1,500 characters** (was 6000; see §1) |
| Chunk overlap | **10–15%** — pick one value; R1 asked for chunking parameters in full |
| Chunk metadata | `chunk_id`, `doc_id`, character offsets |
| Chunk store | `chunks.jsonl`, one per corpus, canonical for both arms |
| Chunks per document | **Measure and report.** Sets the ceiling on $k_{\mathrm{doc}}$ |

### 10.3 Method — generation

| Parameter | Value |
|---|---|
| Granularity $N$, Repair | 1, 5, 10, 20, 54 |
| Granularity $N$, Federation | 1, 5, 10, 23 |
| Headline granularity | $N = 10$ |
| Batching | Contiguous slices of $\pi$, size $\lceil n/N \rceil$, never randomised |
| Carry-forward | Full serialisation of $FM^{(j-1)}$, not a summary |
| Grounding conditions | RAG, Non-RAG |
| Models (open-weight) | `glm-5.3-flash` (Z AI, index 42, 1M ctx), `deepseek-v4.1-flash` (DeepSeek, index 40, 1M ctx) |
| Model (proprietary) | GPT-6 Astra (OpenAI, index 51, 1M ctx) |
| Reasoning effort | **high** for GPT-6 Astra; one fixed setting per open model, recorded in D10 |
| Temperature | 0.2 where the model exposes it; reasoning effort is the operative knob for these models |
| Model snapshot date | Selection made from Artificial Analysis Intelligence Index v4.3, 20 September 2026. Record exact model strings and dates. |
| Max output tokens, open-weight | 16,384 |
| Max output tokens, proprietary | 32,768 |
| Serving, open-weight | Ollama cloud; avoid 12:00–18:00 UTC weekdays (DeepSeek peak pricing doubles) |
| Prompt block order | Invariant blocks first (system, metamodel, output contract), variable last (context, previous model), for prefix caching |
| Stop reason | **Logged every call.** `finish_reason: "length"` marks the run truncated; flag, do not score |
| Seed | Per run, recorded in D10 |
| Repetitions, headline arms | 20 |
| Repetitions, curve arms | 5 |
| Repetitions, $k$ sweep | 3 |
| Metamodel block | Present; removed in the ablation arm |
| Orchestration | LangChain |

### 10.4 Method — retrieval (RAG arms only)

| Parameter | Value |
|---|---|
| Embedding model | `nomic-embed-text`, served by Ollama, version pinned in D12 |
| Embedding prefixes | `search_document:` on chunks, `search_query:` on queries — mandatory |
| Vector store | ChromaDB, one collection per corpus |
| Index construction | Derived from `chunks.jsonl`; `id = chunk_id`, metadata carries `doc_id` |
| Similarity | Cosine |
| Queries | Four fixed sub-queries, identical at every step and in every RAG arm, stored and hashed in D6 |
| Per-sub-query depth | $k_{\mathrm{step}}/4$, merged and deduplicated by `chunk_id` |
| Batch scoping | Metadata filter `doc_id ∈ B_j` |
| $k_{\mathrm{doc}}$ | **TBD**, must be strictly below chunks-per-document |
| $k_{\mathrm{step}}$ | $k_{\mathrm{doc}} \times \lvert B_j \rvert$ |
| $k_{\mathrm{doc}}$ sweep | 3, 5, 10, 15 (Repair, $N=10$, 3 repetitions) |
| Reranking | None |
| Query rewriting | None |

### 10.5 Method — candidate selection

| Parameter | Value |
|---|---|
| Admissibility | Conformance and satisfiability both required; failures discarded, not down-ranked |
| Score | $\alpha \cdot \mathrm{F1} + \beta \cdot \mathrm{cov}$ |
| $\alpha, \beta$ | 0.6, 0.4 |
| Tie-break order | score, then F1, then coverage, then generation order negated |
| $f_m$, models retained | **TBD** — the old paper wrote top-$f_m$ without fixing the value |
| Timing | Strictly post hoc; invisible to generation |

### 10.6 Instrument — semantic measures

| Parameter | Value |
|---|---|
| Matching embedding | `all-mpnet-base-v2`, version pinned in D12 |
| Similarity | Cosine |
| Threshold $\tau$ | 0.4 |
| $\tau$ sweep | 0.3, 0.4, 0.5, 0.6 (free rescore of D15) |
| Stored in D15 | Similarity score, **not** the match boolean |
| Recall targets | $\mathrm{reach}(T,C)$ primary, $F_t$ as labelled bound, $F_t^{att}$ and $F_t^{org}$ separately |

### 10.7 Instrument — structural measures

| Parameter | Value |
|---|---|
| Primary | Parent-match |
| Shape | Feature count, tree depth, branching factor, against the reference |
| Secondary | Semantic coverage, reported beside its signals, never in place of them |
| $w_f, w_p$ | 0.8, 0.2 (design choice, no sweep, non-load-bearing) |

### 10.7b Instrument — variability measures

Nothing in the original design scored `mandatory` or group kind, so a model that recovered every name and every parent, marked everything optional and made every group `and`, would pass all three criteria while being a labelled tree rather than a feature model. These two measures close that hole.

**Reference-free (both corpora).** Computed from the generated model alone, so Federation participates fully despite its encoding.

| Measure | Definition |
|---|---|
| Group-kind distribution | Proportion of internal nodes that are `and`, `or`, `alt`. Proportions, not counts, since models differ in size |
| Mandatory ratio | Proportion of non-root features carrying `mandatory="true"` |
| **Degenerate flag** | True when a model has no `or`, no `alt` and no mandatory features. Report the **share of degenerate models per configuration** |

The degenerate share is the headline number here. A degenerate model passes conformance, is trivially satisfiable, has no dead features, and can score well on names and parents, so a substantial share demonstrates the paper's title rather than arguing it.

**Reference-based (Repair; Federation only after the group-kind fix below).**

| Measure | Definition |
|---|---|
| Group-kind agreement | For each matched feature internal in both models, does the group kind match? Report accuracy **and the 3×3 confusion matrix**; the expected failure is collapse to `and` |
| Mandatory agreement | For each matched non-root feature, does the mandatory flag match? Report accuracy and the direction of error |

**Prerequisite.** `federation_fixed.xml` currently has 33 `and` groups and zero `or` or `alt`, so it carries no variability semantics and the reference-based measures cannot run on it. Fix the four obvious exclusives — `Arity` (Binary / N_ary), `Exec_mode` (Incremental / FullRebuild), `Reified` (Explicit / Implicit), `Trigger` (Manual / Reactive) — as `<alt>`, **before any runs**, and record the change in the artefact. Done beforehand it is correcting an encoding error; done after seeing results it is tuning.

### 10.8 Instrument — logical measures

| Parameter | Value |
|---|---|
| Conformance | Schema validity including key/keyref, plus W1–W5 checked separately |
| Violation reporting | By condition name, not a single boolean |
| Independent parse | FeatureIDE, reported separately from conformance |
| Satisfiability | SAT over the propositional encoding |
| Solver | **TBD** — name and version; FeatureIDE's Sat4j if used through the toolchain |
| Dead features | Reported per model |
| Constraint recovery | Count and type against the reference; Repair only |

### 10.9 Instrument — provenance

| Parameter | Value |
|---|---|
| Emission requirement | Every generated feature names the `doc_id`s it derives from |
| Marker grammar | **TBD** — the literal format the prompt demands, and its parser |
| L0 | Emission compliance: parseable attribution present |
| L1 | Referential integrity: identifier names a corpus document |
| L2 | Context availability: identifier was in context at `first_seen_step` |
| Applicability | L2 undefined for Non-RAG at $N=1$; report as a property |
| Recency check | Citation distribution against batch position |

### 10.10 Instrument — statistics

| Parameter | Value |
|---|---|
| Central tendency | Mean with 95% confidence interval over repetitions |
| Dispersion | Standard deviation per configuration |
| Comparison test | Mann-Whitney $U$ |
| Correction | Holm, across the whole family of tests |
| Effect size | Cliff's $\delta$, reported alongside adjusted $p$ |
| Inter-rater agreement | Quadratic weighted Cohen's $\kappa$ |
| Expert vs Top-FM | Rank correlation |

### 10.11 Infeasibility

| Parameter | Value |
|---|---|
| Context window per model | **TBD** — look up and record per model and version |
| Policy | If assembled context exceeds the window, record the run infeasible |
| Forbidden | Silent truncation |
| Logged always | Assembled token count, in D10, for every run including feasible ones |

---

## 11. Decisions still open

Eight values are not yet fixed. The first four block the campaign; the rest block analysis and can be settled while runs proceed.

**Blocking**

1. **Chunk size and overlap.** Settle on a value in 1,200–1,500 characters and an overlap in 10–15%. Both appear in the paper regardless, since R1 asked for chunking parameters in full. Changing this later invalidates any index already built.
2. **$k_{\mathrm{doc}}$.** Cannot be chosen before chunks-per-document is measured, and that figure roughly quadruples under the new chunk size, so the old grid does not carry over. Must stay strictly below chunks-per-document, or RAG degenerates into Non-RAG at $N = n$. Rescale so a step receives a comparable amount of text to before.
3. **Max output tokens — set, but verify.** 16,384 open-weight and 32,768 proprietary. Late-step output on Repair at $N = 54$ is estimated at 10,000–16,000 tokens, so the open-weight cap is adequate but not comfortable. Run one $N = 54$ arm on one open-weight model (54 calls) and check the final output size before the campaign. The asymmetry is also a confound for the per-LLM comparison: if open-weight arms truncate and proprietary ones do not, part of any conformance gap is budget rather than capability. Log `finish_reason` and report truncation rate per model.
4. **Provenance marker grammar.** It lives in the frozen prompt, so it cannot change after the campaign starts.

**Non-blocking**

5. **$f_m$**, how many top models are retained for reporting.
6. **SAT solver** name and version.
7. **Context window per model and version**, needed to apply the infeasibility policy consistently.
8. **Expert rubric**, the wording raters score against. Start early; raters are the long pole.
