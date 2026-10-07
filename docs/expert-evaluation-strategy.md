# iFS 2027 expert evaluation: lean blinded study (protocol v3)

**Protocol version:** 3.0, 2 October 2026. **Replaces:** v1.0 (29 September 2026), the
blinded three-model preference design. **Status:** preparation. Do not dispatch until
the gate in §6 passes.

**Documentation revision (7 October 2026):** removed the administrative schedule
and clarified preparation inputs. The selection rule and analysis plan are unchanged.

This study is an exploratory amendment designed after the automated results were
known. It is reported as such and not as a prospectively registered comparison.
Freeze this file, the workbook, the selection rule and the packet scripts before any
rater sees material, and record their SHA-256 hashes.

## 1. What the study is for

The paper argues that the verdicts of reference-based evaluation depend on unstated
conventions. A human study is useful to the paper only if it tests that argument, so
this one asks no "which model is best" question. Its three questions map onto the
paper's research questions:

| ID | Question | Supports | Measured by |
| --- | --- | --- | --- |
| **E1** | When experts rank the models of one corpus, is their ranking closer to the ranking under one-to-one matching or under independent-max matching? | RQ2 (matching convention) | Task 2 ranks against the two automated rankings |
| **E2** | Do experts judge degenerate models, which pass every automated gate, as poorer in variability and less acceptable? Do they judge schema-invalid but loadable models differently from conformant ones? | RQ1 (cheap gates) | Task 1 Q4 and the overall decision, against degeneracy and XSD status |
| **E3** | Does a cited document support the feature that cites it, and does the answer differ between citations the reference attribution agrees with and citations it does not? | RQ4 (provenance) | Task 3 support judgements by hidden stratum |

A secondary, descriptive question: **do experts penalise redundancy** (Q5) where
independent-max rewards it, measured as the relation between Q5 and the generated-to-
reference size ratio.

Out of scope: a model leaderboard, the metamodel ablation, and any claim that a
difference between models is caused by model weights. Providers, reasoning settings and
temperatures differ, and the sample has one model per cell.

## 2. Sample

### 2.1 Cells

Per corpus, five cells, all guided, RAG, primary ordering, `k_doc = 5`:

| Cell | Source | Why it is in the sample |
| --- | --- | --- |
| DeepSeek, `N = 1` | 694-run campaign | compact; often degenerate |
| GLM, `N = 1` | 694-run campaign | compact; rarely degenerate |
| DeepSeek, `N = 10` | 694-run campaign | over-produced (about 2–3× the reference) |
| GLM, `N = 10` | 694-run campaign | close to the reference size |
| Astra, `N = 1` | expert extension `expert-astra-n1-rag-v1` | over-produced at a size experts can still read (1.5–2×) |

Astra at `N = 10` (about 600 features) is excluded because it is not legible in a
reasonable time. The Astra `N = 1` runs are a labelled extension. They are not part of
the 694-run campaign, are scored under the same evaluation contract, and appear in the
paper only in the expert analysis.

The five cells span the property that drives the two matching policies apart, output
size. On the selected sample the policies disagree about the best model in both
corpora (§2.3), so the experts' ranking can discriminate between them.

### 2.2 Selection rule (frozen)

For each corpus, select the **lowest seed in {0, …, 4} for which all five cells have a
completed, syntactically extractable output**, and take that seed's output from every
cell. Do not condition on schema validity, satisfiability, degeneracy or any score.
Do not repair or edit any output. If a selected output cannot be rendered, move to the
next qualifying seed for the **whole corpus** and record why.

Applying the rule gives seed 0 in both corpora (seeds 0, 1, 3, 4 qualify for Repair and
0, 3 for Federation). The rule is implemented by `prepare_expert_artifacts.py select`, which writes
the author-only key `expert_sample_frozen.csv` and its SHA-256; record the hash here
before any rendering.

**Neutral codes.** The implementation preserves the author-approved `Rep-NN` and
`Fed-NN` mappings in §2.3 rather than drawing new codes. Record this as the implemented
coding rule; do not claim a new random draw was performed by the preparation script.
The domain prefix
reveals nothing raters cannot already see; the number carries no information about
model, `N`, seed or run. All raters see the same codes; only the presentation order
differs per rater. The code-to-run key stays author-only until the data are locked.

### 2.3 Frozen sample (10 models)

Automated values are author-only and never shown to raters.

| Corpus | Cell | Code | Run ID | Features | Size ratio | XSD valid | Loads in FeatureIDE | Degenerate | F1 ind. | F1 1:1 | Rank ind. | Rank 1:1 |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | :-: | :-: |
| Repair | DeepSeek `N=10` | Rep-28 | `76eae3eb6c3292c4` | 309 | 3.09 | no | yes | n/a | 0.717 | 0.342 | 1 | 3 |
| Repair | Astra `N=1` | Rep-79 | `92ea4c2daeb9f48f` | 215 | 2.15 | yes | yes | no | 0.682 | 0.444 | 2 | 2 |
| Repair | GLM `N=10` | Rep-57 | `6fb3c9722baab5e3` | 119 | 1.19 | yes | yes | no | 0.631 | 0.466 | 3 | 1 |
| Repair | DeepSeek `N=1` | Rep-96 | `2b109d27ae701f4a` | 30 | 0.30 | yes | yes | no | 0.450 | 0.338 | 4 | 4 |
| Repair | GLM `N=1` | Rep-30 | `7f08e9c1343fb8bd` | 15 | 0.15 | no | yes | n/a | 0.423 | 0.226 | 5 | 5 |
| Federation | DeepSeek `N=10` | Fed-97 | `449b541af841413a` | 248 | 1.91 | yes | yes | **yes** | 0.594 | 0.339 | 1 | 3 |
| Federation | Astra `N=1` | Fed-44 | `8b15740d63e44585` | 199 | 1.53 | yes | yes | no | 0.580 | 0.395 | 2 | 1 |
| Federation | GLM `N=10` | Fed-67 | `6dab8574ea5f471f` | 136 | 1.05 | yes | yes | no | 0.528 | 0.376 | 3 | 2 |
| Federation | GLM `N=1` | Fed-11 | `2eb5d9748a396e07` | 27 | 0.21 | yes | yes | no | 0.347 | 0.255 | 4 | 4 |
| Federation | DeepSeek `N=1` | Fed-36 | `0ec82d2172bb7fe3` | 21 | 0.16 | yes | yes | **yes** | 0.344 | 0.238 | 5 | 5 |

Degeneracy is undefined for schema-invalid outputs in the campaign data; compute it on
the parsed tree for the two Repair cases before analysis and record the value.

The sample also contains, without any selection for it, two degenerate models and two
schema-invalid models that load in FeatureIDE. These are the cases E2 needs.

## 3. Instrument (workbook v3)

`Expert-evaluation-form.xlsx` (renamed from the v3 filename). All input cells are yellow and constrained by
drop-down lists. Model identities, run IDs, scores and reference models never appear.

**Task 1. Rate each model (10 rows).** Rows are grouped by corpus. The order of the two
corpus blocks is counterbalanced across raters, and the order of models within a block
is randomised per rater. For each model: Q1 present-concept fidelity, Q2 important-
concept coverage, Q3 hierarchy, Q4 variability and constraints, **Q5 redundancy (new)**,
each on 1–5 or `NJ`; an overall first-draft decision (Accept, Accept with revision,
Reject, Cannot judge); one wrong and one missing concept if identified; minutes spent.
Q1 no longer includes redundancy, so that Q5 measures it separately.

**Task 2. Rank within each corpus (2 × 5 rows).** Only after Task 1, rank the five models
of each corpus from 1 (best first draft) to 5, with ties allowed, and give a one-line
reason for the best and the worst. This replaces the v1 three-way choices and is the
primary outcome for E1.

**Task 3. Check citations (12 items).** Each item shows one feature, its hierarchy path,
the cited document and a short excerpt. The rater judges whether the evidence supports
the feature: Yes, Partially, No, Cannot judge. The hidden stratum of each item is
author-only (§4).

**Removed from v1:** the eight three-way choices, the two repeat presentations and the
26-row layout.

**Estimated burden:** about 95 minutes for Task 1 (compact models about 5 minutes,
larger ones 10 to 15), 10 minutes for Task 2 and 35 minutes for Task 3. About **2 h 20 min**
in total, with breaks. Confirm in the pilot (§6).

## 4. Citation items

Pool: every citation edge of the ten selected models whose feature has an
independent-max match (τ = 0.4) to an **attested** reference feature. Label each edge
*agree* if the reference attributes the matched feature to the cited document, and
*disagree* otherwise.

Draw, per corpus, **3 agree and 3 disagree** edges uniformly at random from the pooled
edges of that corpus's five models, with at most two items per model and a fixed seed
(`20261002`). Excerpts are chosen by a researcher from the cited document or from the
logged retrieval chunk, **before** the stratum is looked up, and a second researcher
checks source identity. Never replace an item because its excerpt does or does not
support the feature. Existing edges per selected campaign model range from 4 to 104 in
each stratum, so the draw is feasible; take the Astra edges from the extension outputs.

## 5. Raters and presentation

- **Raters:** two at minimum, three if available. They need working knowledge of model
  repair, model federation or both; a rater who cannot judge a corpus uses `NJ`.
  Record consent and data handling as required locally. Use anonymous rater codes.
- **Presentation:** one identical, searchable HTML tree view for every model, with
  collapsible subtrees, the mandatory and optional markers, `or` and `alt` groups and
  cross-tree constraints, feature descriptions with trace markers stripped, and no
  provider, run or size information. Source documents (or authorised links) and a
  document index are supplied for optional checking; reference models are not.
- **Instructions:** "Do not infer quality from size alone." Raters work independently
  and do not discuss the study before all workbooks are returned.

## 6. Preparation and replication

### Required inputs

The software repository provides the preparation scripts, not all study outputs.
`results/` is intentionally Git-ignored: obtain the compatible **research-data
archive** and restore its files to their original paths using
[the reproduction guide](reproduce.md). Git-ignore prevents accidental commits;
it does not prevent restored files from being used by the scripts. The data
record is currently a draft; until a public DOI/version is recorded, request the
compatible archive from the authors. Do not assume every expert-study input is
already included in the non-expert release.

| Input | Expected location after restoration | Purpose |
| --- | --- | --- |
| Main-campaign inventory and saved run artefacts | `results/ifs-2027/analysis/inventory-current-v1/runs.json` and the campaign paths it identifies | Select the eligible campaign outputs |
| Saved Astra extension runs | `results/expert-astra-n1-rag-v1/` | Select the two extension outputs without new model calls |
| Main semantic pair scores | `results/ifs-2027/analysis/semantic-current-v3/pairs.csv` | Form the citation sampling pools |
| Reference XMLs, attribution and document manifests | Original `data/` paths | Resolve reference matches and source identities |
| Pinned evaluation encoder in the local cache, or saved extension pairs | See [semantic evaluation](semantic-evaluation.md); alternatively use `--astra-pairs PATH` | Score selected Astra features locally |
| Reviewed blank instrument | `data/Expert-evaluation-form.xlsx` | Build the rating workbooks; never substitute a returned expert form |
| Checked citation excerpts and source locators | A separately completed citation-items CSV | Populate the evidence shown to raters |

Original source papers or authorised access may be needed to verify excerpts;
their full texts are not guaranteed to be redistributed. Exact replication of
the **dispatched packets**, rather than just model selection, requires the frozen
selection, completed citation items, approved instrument, presentation settings
and packet hashes. These must be released subject to rights and blinding review.
If they are unavailable, state that limitation rather than inventing excerpts or
claiming the packets were reproduced exactly.

### Preparation procedure

Run from the repository root after installing the release environment and
restoring the inputs. Use new output directories, as below, to preserve the
original study artefacts. These commands make no generation/API calls.

**1. Select and freeze the sample.** Apply §2.2 and retain the selection hashes.

```bash
./.venv-release/bin/python scripts/prepare_expert_artifacts.py select \
  --output results/expert-study/selection-reproduced-v3
```

The default inputs are listed above; override them with `--open-inventory` and
`--astra-results` when needed. The selection directory contains the model-code
key and remains author-only until responses are locked. Selection checks frozen
configurations except `N`, not semantic scores or XSD outcomes.

**2. Sample and verify citation evidence.** Draw items under §4, then have the
excerpts and source identities independently checked.

```bash
./.venv-release/bin/python scripts/prepare_expert_artifacts.py citations \
  --selection-dir results/expert-study/selection-reproduced-v3 \
  --output results/expert-study/citations-reproduced-v3
```

Use `--pairs PATH` for a different restored main pairs location. The Astra summary
CSV is not a raw pairs table: without `--astra-pairs`, the command computes the
selected Astra similarities locally with the pinned encoder. It requires valid
same-corpus document IDs and attested reference matches at τ = 0.4, and preserves
occurrence indices to distinguish duplicate feature paths.

Sampling uses uniform three-edge subsets in each corpus/stratum, rejecting the
whole six-edge draw when any selected model contributes more than two edges.
This is uniform **conditional on the cap**, not unconstrained equal marginal
inclusion probability for every edge. The fixed seed is 20261002. Insufficient
pools or failure to find a valid draw stop rather than relax the sampling rule.

Give the excerpt preparer only `citation_items_to_complete.csv`. Keep
`citation_candidates_AUTHOR_ONLY.csv`, `citation_key_AUTHOR_ONLY.csv` and
summary.json private: they reveal strata and model/run mappings. Preserve the
blank CSV; save a separate `citation_items_final-reproduced-v3.csv`, filling only:

- excerpt
- excerpt_origin (`retrieved chunk` or `document excerpt`)
- excerpt_locator
- source_checked_by (second researcher's initials/code; not sent to raters)

Do not change item IDs, codes, domains, features or document identities. All 12
are required. The excerpt preparer must not inspect the private strata first.

**3. Assemble and inspect packets.** Use the checked citation file and a reviewed
blank form compatible with the packet builder.

```bash
./.venv-release/bin/python scripts/prepare_expert_artifacts.py pack \
  --selection-dir results/expert-study/selection-reproduced-v3 \
  --citation-dir results/expert-study/citations-reproduced-v3 \
  --citation-items results/expert-study/citation_items_final-reproduced-v3.csv \
  --form data/Expert-evaluation-form.xlsx \
  --output results/expert-study/packets-reproduced-v3
```

The default is three raters; use `--raters 2` for two. Corpus block order alternates;
model and citation orders are seeded per rater. Keep `AUTHOR_ONLY` outside the
dispatched packets. The builder removes comments from packet copies, not the
source form, and rejects incomplete evidence or incompatible templates. In
particular, the current ten-column citation layout requires verified feature
descriptions and source links, and extra citation-sheet rows must be reviewed in
a copy of the blank instrument. The four excerpt fields alone do not guarantee
that every instrument revision can be packed; do not bypass validation failures.

**Dispatch gate:** inspect all ten rendered models, ten rating rows, two five-model
ranking blocks and twelve citation items. Verify source excerpts, blank answer
cells and absence of model identities or hidden keys. Pilot one compact and one
large model with a non-study reader, version any wording changes, and record
the final packet hashes. Automated checks do not replace this inspection.

**4. Lock and analyse responses.** Validate returned values, preserve originals
privately, prepare de-identified analysis records, lock the data and only then
unblind. Follow §7. If fewer than two complete workbooks are available, report
that limitation and do not present the planned multi-rater analysis as complete.

## 7. Analysis plan (frozen with this protocol)

All analyses are descriptive. Small samples, no corrected significance claims.

- **E1.** For each rater and corpus, Kendall's τ between the rater's ranking and (a) the
  one-to-one ranking and (b) the independent-max ranking. Report all values and how
  many rater–corpus pairs favour each policy. Report inter-rater agreement on the
  rankings with Kendall's W.
- **E2.** Q4 and the overall decision for the two degenerate models against the eight
  others, and for the two schema-invalid models against the conformant ones. Report
  each model's ratings; do not average over two models and call it an effect.
- **E3.** Yes / Partially / No / Cannot judge counts by stratum (agree, disagree) and by
  corpus. If disagree items are judged supported about as often as agree items, the
  chance-level agreement in RQ4 reflects incomplete reference attribution rather than
  unsupported citations; if they are not, it reflects citations that carry little
  evidence. Report whichever is found.
- **Secondary.** Spearman correlation between Q5 and the size ratio, and Q1–Q4 by
  model, with weighted κ (quadratic) between raters per criterion.
- `NJ` and `Cannot judge` are excluded from each statistic, never coded as zero, and
  their counts are reported.

## 8. Changes from v1.0

| v1.0 | v3.0 | Reason |
| --- | --- | --- |
| Primary outcome: blinded three-way model preference | Primary outcome: within-corpus ranking compared with each matching policy | The paper is about evaluation conventions, not model ranking |
| `N = 1` only | `N = 1` and `N = 10` | At `N = 1` the two matching policies largely agree, so they cannot be told apart |
| 24 models, 8 triplets, 2 repeats, 15 citation items | 10 models, 2 rankings, 12 citation items | Reduced estimated burden from about 4–5 h to about 2 h 20 min |
| Citation items balanced by model | Citation items stratified by hidden agreement with the reference | Tests the RQ4 finding directly |
| Q1 includes redundancy | Q5 redundancy separate | Tests whether experts penalise what independent-max rewards |
| Astra `N = 1` extension as a model in a leaderboard | Astra `N = 1` as the readable over-producer | Same runs, different role |

Unchanged from v1.0: blinding, neutral codes, a single renderer, no editing of outputs,
`NJ` handling, the attrition and deviation logging, the consent and data-release
rules.
