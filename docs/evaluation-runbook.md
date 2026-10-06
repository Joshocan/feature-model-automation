# IFS 2027 offline evaluation handoff

These commands read existing campaign outputs. They do **not** invoke OpenAI,
Ollama, or any generation campaign. Run them manually from the repository root,
after writers have stopped. Every `--output` must be a **new** directory; keep
older snapshots. The old `inventory-2026-09-26-v2` has the previous evaluation
contract hash, so do not reuse it after the D01 decision.

First verify that the matching encoder's exact recorded revision is locally
available (D03 in `evaluation-decisions.md`). `evaluate_semantic.py` refuses a
different revision or dependency identity; do not substitute a current model.
The FeatureIDE application parser, final strict-sensitivity gate, inferential
families, and human-audit scope are still open decisions (D02/D04/D05).

```bash
./.venv/bin/python -m pip install -r config/requirements.txt

./.venv/bin/python scripts/inventory_campaign.py \
  --output results/ifs-2027/analysis/inventory-current-v1

./.venv/bin/python scripts/evaluate_structure.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/structure-current-v1

./.venv/bin/python scripts/evaluate_semantic.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/semantic-current-v1

./.venv/bin/python scripts/evaluate_provenance.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/provenance-current-v1

./.venv/bin/python scripts/aggregate_campaign.py \
  --structural results/ifs-2027/analysis/structure-current-v1 \
  --semantic results/ifs-2027/analysis/semantic-current-v1 \
  --provenance results/ifs-2027/analysis/provenance-current-v1 \
  --output results/ifs-2027/analysis/aggregate-current-v1

./.venv/bin/python scripts/verify_evaluation_outputs.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --structural results/ifs-2027/analysis/structure-current-v1 \
  --semantic results/ifs-2027/analysis/semantic-current-v1 \
  --provenance results/ifs-2027/analysis/provenance-current-v1 \
  --aggregate results/ifs-2027/analysis/aggregate-current-v1
```

The verifier checks that all three metric stages and the wide table account for
every planned run exactly once, and prints completed and eligible denominators.
It does not certify publication validity. Inspect non-`ok` status counts and
`summary.json` files, particularly any encoder or parser limitations.

After D04 families are declared in versioned JSON files, run each with
`scripts/analyse_family.py --wide .../wide.csv --family ...json --output ...`.
Use `semantic_population: "completed_extractable_final"` for the main semantic
comparison and a separate declaration with `"strict_admissible_final"` for
the sensitivity analysis. The family driver and boxplot both apply the same
arm filter, population gate, and `status=ok` rule; low-n cells remain labelled.

For the τ sensitivity, run `scripts/tau_rescore.py --pairs
.../semantic-current-v1/pairs.csv --output .../tau-current-v1`. This uses the
occurrence-indexed pairs from the semantic evaluator.
The following dry run checks plotting inputs without writing figures:

```bash
./.venv/bin/python scripts/plot_figures.py \
  --campaign results/ifs-2027/analysis/aggregate-current-v1 \
  --tau results/ifs-2027/analysis/tau-current-v1 \
  --wide results/ifs-2027/analysis/aggregate-current-v1/wide.csv \
  --output results/ifs-2027/analysis/plots-current-v1 \
  --plot degenerate_share n_curve criterion_heatmap tau_sweep --dry-run
```

Remove `--dry-run` only after checking the reported cell/sample counts. For
`family_boxplot`, also supply `--family` pointing to a completed family output.
Publication mode fails on missing or empty inputs; `--allow-placeholder` is for
exploration only and should not be used to generate paper figures.
