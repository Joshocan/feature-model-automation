# iFS 2027 expert evaluation: lean blinded study (protocol v3)

**Protocol version:** 3.0, 2 October 2026. **Replaces:** v1.0 (29 September 2026), the
blinded three-model preference design. **Status:** preparation. Do not dispatch until
the gate in §6 passes. **Submission deadline:** 15 October 2026.

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

`Expert-evaluation-form-v3.xlsx`. All input cells are yellow and constrained by
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

## 6. Steps and timeline

### Implemented v3 preparation commands

Run from the repository root. Each output directory must be new. These commands
make no generation/API calls. The v1/v2 commands in the older strategy document
are superseded. `build_expert_form.py` still reproduces the historical v2 form;
do not use it to overwrite the supplied v3 workbook.

1. Freeze the ten selected models and approved neutral codes:

```bash
./.venv/bin/python scripts/prepare_expert_artifacts.py select \
  --output results/expert-study/selection-v3
```

Defaults: campaign inventory-current-v1/runs.json and
results/expert-astra-n1-rag-v1. The entire selection directory is author-only.
It includes selection.json, expert_sample_frozen.csv, astra_attempts.csv and
hashes.json. Selection compares frozen configurations except N (which deliberately
differs), not semantic scores or XSD outcomes.

2. Draw citation items under §4:

```bash
./.venv/bin/python scripts/prepare_expert_artifacts.py citations \
  --selection-dir results/expert-study/selection-v3 \
  --output results/expert-study/citations-v3
```

The default main pairs table is semantic-current-v3/pairs.csv. The Astra one-CSV
export does not contain raw pair matrices, so this command computes similarities
locally for the two selected Astra models using the pinned cached encoder. This
can take time but involves no API calls. If raw extension pairs already exist,
pass `--astra-pairs PATH`. It requires valid same-corpus document IDs and attested
reference matches at tau=.4, excludes other edges, and keeps occurrence indices
privately to disambiguate duplicate feature paths.

Sampling uses uniform three-edge subsets in each corpus/stratum, rejecting the
whole six-edge draw when any selected model contributes more than two edges.
This is uniform **conditional on the cap**, not unconstrained equal marginal
inclusion probability for every edge. The fixed seed is 20261002. Insufficient
pools or failure to find a valid draw stop rather than relax the sampling rule.

Give the excerpt preparer ONLY `citation_items_to_complete.csv`. Keep
`citation_candidates_AUTHOR_ONLY.csv`, `citation_key_AUTHOR_ONLY.csv` and
summary.json private: they reveal strata and model/run mappings. Preserve the
blank CSV; save a separate `citation_items_final-v3.csv`, filling only:

- excerpt
- excerpt_origin (`retrieved chunk` or `document excerpt`)
- excerpt_locator
- source_checked_by (second researcher's initials/code; not sent to raters)

Do not change item IDs, codes, domains, features or document identities. All 12
are required. The excerpt preparer must not inspect the private strata first.

3. Build packets after evidence completion and second-researcher checks:

```bash
./.venv/bin/python scripts/prepare_expert_artifacts.py pack \
  --selection-dir results/expert-study/selection-v3 \
  --citation-dir results/expert-study/citations-v3 \
  --citation-items results/expert-study/citation_items_final-v3.csv \
  --form data/Expert-evaluation-form-v3.xlsx \
  --output results/expert-study/packets-v3
```

Default three raters; `--raters 2` creates two. Each ZIP contains ten collapsible
HTML models, the workbook with ten rating rows, two five-model ranking blocks,
twelve citation rows, a source index and instructions. No triplets or repeats.
Corpus block order alternates across raters; model order and citation order are
seeded per rater. Citation judgement/reason cells stay blank. Prior responses,
hidden/extra sheets, changed frozen files or changed sampled identities are
rejected. Comments (including author metadata) are removed from packet copies,
not from the source workbook. A/B/C choice sheets and Rater details from v2 are not used.
Keep AUTHOR_ONLY private and run the pilot/dispatch gate below: machine checks
do not replace independent inspection of source text, rendering and blinding.

| Date | Step |
| --- | --- |
| Thu 2 Oct | Freeze this protocol, the workbook and the selection output; record hashes. |
| Fri 3 Oct | Draw the citation items (§4); a researcher selects excerpts, a second checks them. Build the packets: the existing `pack` command with a 10-row Task 1, a ranking sheet and 12 citation rows. |
| Fri 3 Oct | **Pilot** with one non-study reader on one compact and one large model. Fix wording, then version the workbook. |
| Sat 4 Oct | **Dispatch gate:** each packet has 10 model files matching 10 Task 1 rows, both corpus blocks, 5 + 5 ranking rows whose codes appear in Task 1, 12 citation rows and no identifying information. Record packet hashes. Send. |
| Thu 9 Oct | Responses due. Validate values, lock the data, then unblind. |
| Fri 10 – Sat 11 Oct | Analysis (§7); write §5.5 (half a page) and update the abstract. |
| Mon 13 – Tue 14 Oct | Final pass. |

**Fallback:** if fewer than two complete workbooks arrive by 9 October, submit without
§5.5, restrict the claims to reference agreement, and describe the study as ongoing
under Threats to Validity.

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
| 24 models, 8 triplets, 2 repeats, 15 citation items | 10 models, 2 rankings, 12 citation items | Burden from about 4–5 h to about 2 h 20 min, feasible before 15 October |
| Citation items balanced by model | Citation items stratified by hidden agreement with the reference | Tests the RQ4 finding directly |
| Q1 includes redundancy | Q5 redundancy separate | Tests whether experts penalise what independent-max rewards |
| Astra `N = 1` extension as a model in a leaderboard | Astra `N = 1` as the readable over-producer | Same runs, different role |

Unchanged from v1.0: blinding, neutral codes, a single renderer, no editing of outputs,
`NJ` handling, the attrition and deviation logging, the consent and data-release
rules.
