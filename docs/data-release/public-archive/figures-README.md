# Paper figures

Use `publication/` for the current six figures. Each figure has PDF, PNG and CSV
files. `figure_manifest.json` identifies its inputs and output hashes. The
inflation annotation uses Spearman r_s. No figure was recalculated during archive
preparation; these are the saved updated outputs.

| File stem | Content |
| --- | --- |
| `fig_inflation` | Independent-max/one-to-one precision ratio against output/reference size |
| `fig_granularity` | Guided-RAG F1 across N, both matching policies, with completed/planned counts |
| `fig_closure_tau` | Weak and strong closure checks by threshold, corpus and matching policy |
| `fig_variability` | Group-kind proportions, mandatory ratio and degeneracy |
| `fig_outcomes` | Guided-arm recorded outcomes |
| `fig_ordering` | Order-sensitivity one-to-one F1 |

`paper-figures-v4/` is preserved historical evidence, not the preferred figure
set. Historical manifests contain original software paths; consult the root
artifact manifest to locate their archived equivalents. Regeneration commands
are in `protocol/reproduction/README.md`. Compare CSV data rather than PDF binary
identity, because rendering metadata can change.
