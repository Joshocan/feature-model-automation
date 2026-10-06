# Paper figures from saved analysis

Run from the repository root (no API calls or new semantic evaluation):

```bash
./.venv/bin/python scripts/plot_paper_figures.py \
  --output results/ifs-2027/analysis/paper-figures-v1
```

Use a fresh output directory for each version. Defaults consume aggregate-current-v4,
inventory-current-v1, tau-matching-v2 and semantic-current-v3 under the analysis
directory. `--analysis` can point to another root with the same versioned layout.
Requires the existing NumPy, SciPy and Matplotlib dependencies; no pandas.

Produces all six requested PDF names, matching PNG previews, per-figure CSV
source data, and figure_manifest.json (input/output hashes, descriptive statistics,
denominators, scope notes and implementation hash). Old evaluation outputs are
never modified. The full completed population is checked for every threshold and
matching policy; duplicate keys or missing scores fail rather than silently drop.

## Figure definitions and captions

**M1 fig_inflation.pdf** (`fig:inflation`, 0.75 textwidth): pooled completed-run
size ratio versus precision ratio, tau=.4. Decimal log ticks are 0.1, 0.25,
0.5, 1, 2, 4, 8; the reference-size line is labelled and legend is upper left.
Zero one-to-one precision would make
the ratio undefined: excluded IDs are recorded. Spearman is descriptive across
heterogeneous configurations, and the plotted ratios share size-related
components. Do not describe the association as a causal test. Model colour and
marker are redundant cues.

**M2 fig_granularity.pdf** (`fig:granularity`, textwidth): four corpus/model
panels in one 4.8 × 2.2 inch row, guided arms only, RAG, primary ordering,
k_doc=5. Dense two-line x ticks use 5.5pt at this compact final size.
Means of completed
scored runs, not means counting failure as zero. Each categorical tick reports
completed/planned. Zero-completion cells say "no output"; empty cells are gaps;
all tied one-to-one peaks are outlined.
The requested 0.2–0.8 y range is used ONLY if it includes every plotted mean;
otherwise all panels use 0–1 to avoid hiding results. The peak is descriptive,
not a statistical estimate of N-star. No confidence intervals are implied.

**A1 fig_closure_tau.pdf** (`fig:closure`, textwidth): weak flag is full-reference
recall above the annotated closed-reach ratio; strong flag is at least one
matched reference node outside that closed reach. Exact out-of-reach ID sets
and counts are computed from the saved matched_reference_ids per run/tau/policy,
not extrapolated from tau=.4. Integer cardinalities determine weak exceedances
to avoid rounded-rho errors. All 500 completed runs are represented per policy
and threshold. Flags alone do not establish hallucination or falsify a
conditional evidential-closure proposition. CSV contains per-run values;
manifest contains aggregate percentages and denominators.
Four distinct colours identify the closure series: blue for independent-max
weak, green for independent-max strong, orange for one-to-one weak, and purple
for one-to-one strong. Solid/dashed lines identify independent-max/one-to-one;
circles/triangles distinguish weak/strong tests for greyscale readability.
This restores the user's preferred four-colour encoding.

**A2 fig_variability.pdf** (`fig:variability`, textwidth): conformant means
local structural_conformance=True with status=ok, not strict admissibility,
FeatureIDE acceptance or SAT. Group shares pool AND/OR/ALT counts across these
models; mandatory ratio is the median over eligible runs, degeneracy is the percent
of eligible runs flagged. Labels give n and CSV gives metric denominators.
Population is now selected from the exact arm_a/arm_b selectors in
config/analysis/families/C_ablation.json, deduplicated at run level. This means
RAG, primary ordering, k_doc=5; open models at N1/N10 and Astra at N10, both
corpora. Conformant guided/ablated counts are DeepSeek 60/56, GLM 59/40,
Astra 17/16. The previous DeepSeek guided n=97 pooled extra guided conditions
and must not be used for this revised chart. Survivor mixtures still differ,
so use Family C tests for controlled claims. The numeric axis ends at 100;
annotations sit outside it. Caption key: **M = median mandatory ratio;
D = percentage degenerate**. The manifest supplies caption text and selector
hash; the source CSV includes exact run IDs. Missing metrics are not zeros.

**A3 fig_outcomes.pdf** (`fig:outcomes`, 0.75 textwidth): all planned guided
baseline/headline/curve/cross-corpus arms, stacked by recorded terminal outcome.
Unknown/unrepresented outcomes cause an error rather than disappear. Completed
does not mean conformant. Label is "non-RAG". Empty responses stay in the
stacks and CSV but are omitted from the legend; their actual count is provided
in the manifest caption and must be mentioned in the paper caption.
The chart shows outcome frequencies: it cannot alone
show that truncation follows output size or that any provider caused failures.

**A4 fig_ordering.pdf** (`fig:ordering`, 0.6 textwidth): Repair N10 RAG, k_doc=5,
guided; primary comes from headline, alternates from order sensitivity. The
strip plot includes every available one-to-one score, deterministic jitter,
mean line and mean +/- sample SD band (NOT a confidence interval). Labels use
actual scored/planned sizes: do not claim every cell has 4–5 runs. Y-axis is
0.2–0.6; the script refuses out-of-range points instead of silently clipping.
Counts, means
and SD are also in the manifest. Unequal survivor populations are a limitation.

PDF widths are final LNCS widths: 4.8 inches for full-width, scaled before
rendering for smaller figures. Serif 8pt (7pt ticks), 2pt lines, horizontal grey
grid and no top/right spines; dense categorical ticks use 6pt. PNG files are
previews; use vector PDFs in LaTeX. Colours for group/outcome categories are not
model identities; hatch patterns provide a second cue.

Example inclusion:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.75\textwidth]{fig_inflation.pdf}
  \caption{Precision-ratio sensitivity to generated model size. Each point is
  one completed run; the pooled correlation is descriptive.}
  \label{fig:inflation}
\end{figure}
```

Page layout and captions determine whether the appendix fits six pages; the
script does not enforce a manuscript page limit.
