# Expert assessment

Two raters assessed ten generated feature models, five per corpus. The sample
contains DeepSeek and GLM at N=1 and N=10 and Astra at N=1, all guided RAG,
using the frozen selected runs. It is not a selection of the highest-scoring
outputs. The model-code key is now supplied for replication; it reveals model
identities and must not be released while blinding is still required.

## Files

- `inputs/ratings.csv`: 20 rows, one per rater/output. Q1–Q5, overall decision,
  numeric decision code and rank. Four Q5 ratings are missing for Expert B.
- `inputs/citation_judgements.csv`: 12 items with both raters' labels/codes.
  Expert B has two Cannot judge responses, not missing unfinished submissions.
- `inputs/model_selection.csv`: ten codes, model/run identities, software-relative
  XML paths and hashes. Absolute personal paths have been removed.
- `inputs/citation_items.csv`: item identity, model code, document identity,
  feature alignment and sampling stratum. It contains no rater identity.
- `automated-scores/astra-n1.csv`: all ten extension outputs, needed for within-cell
  percentiles, not just the two selected Astra outputs. Personal checkout prefixes
  in its provenance JSON keys are replaced by software-relative paths; numeric
  scores and recorded input hashes are unchanged. The manifest records the source
  file hash and this public-derivative transformation.
- `anonymised-results/`: 20 analysis CSVs and `summary.md`, including the merged
  output-level table, agreement, rank correlations and within-cell percentiles.

The two response CSVs are byte-identical copies of the normalised response-level
exports. No free-text comments, signatures, workbook properties or private
identity mappings are included. Normalisation has already occurred: the public
replication verifies analysis from these inputs, not extraction from withheld forms.

## Definitions

Q1–Q5 are 1–5 ratings. Blank Q5 cells represent NJ, not zero. Overall decisions
are Reject=0, Accept with revision=1, Accept=2. Cannot judge is missing.
Citation coding is Yes=2, Partially=1, No=0; Cannot judge is missing. Expert B's
author-authorised mapping was OK→Yes, Not exactly→Partially, and
"not sure, depends on the context"→Cannot judge.

Ordinal Krippendorff alpha uses domains 1–5 for ratings and 0–2 for decisions and
citations. Rank concordance uses tie-corrected Kendall W. Correlations use
Kendall tau-b, negating expert ranks when larger automatic scores mean better.
Percentiles use the mid-rank definition and the selected output's completed
guided-RAG cell. See the companion software's `docs/reproduce.md`, Section 5.5,
for all estimands and eligibility rules.

## Reproduce without private workbooks

After restoring the archive into the updated software checkout:

```sh
./.venv-release/bin/python scripts/expert_analysis.py \
  --public-inputs results/expert-study/public-inputs \
  --output results/expert-study/analysis-reproduced
```

Use a new output directory. The script checks 82 supplied published values and
the selected XML hashes before writing final tables. Compare its result CSVs
with `anonymised-results/`; input/software hashes differ appropriately because
public CSVs replace private workbooks. The archived public-input verification
records the comparison. All calculations are offline and make no model calls.

These are descriptive findings from two raters and ten outputs. The public
inputs are technically anonymised, but consent/rights review and release of the
model-code key still require researcher sign-off.
