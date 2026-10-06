#!/usr/bin/env python3
"""plotting scaffolding for the paper's headline figures.

Reads the campaign aggregator outputs (``cell_summary.csv``, ``variability.csv``, ``correlations.csv``)
and statistical-pass outputs (``tau_sweep.csv``, ``family_results.csv``) outputs, and writes
publication figures. Every plot is emitted as both PDF and PNG at the same
DPI so LaTeX and screen previews are identical.

**This script is deliberately CLI-only.** Nothing runs unless you invoke
it; there is no auto-discovery.

Plots produced (each is a separate ``--plot`` slug):

  * ``degenerate_share``  — bar chart of degenerate share per cell
                             (variability.csv → axes)
  * ``n_curve``           — mean F1_total vs N per (model × grounding × corpus)
                             from cell_summary.csv
  * ``criterion_heatmap`` — Spearman ρ heatmap between the three criteria
                             from correlations.csv
  * ``tau_sweep``         — mean F1_total vs τ per guided cell, with n
                             from tau_sweep.csv joined to wide.csv
  * ``family_boxplot``    — headline arm distributions per comparison
                             from wide.csv (needs --wide too)

Publication mode fails when a required input is absent. Exploratory mode may
opt in to labelled placeholders with ``--allow-placeholder``.

Usage
-----

    ./.venv/bin/python scripts/plot_figures.py \\
      --campaign results/ifs-2027/analysis/campaign-2026-09-27-v1 \\
      --tau      results/ifs-2027/analysis/tau-sweep-2026-09-27-v1 \\
      --family   results/ifs-2027/analysis/family-h1b-2026-09-27-v1 \\
      --wide     results/ifs-2027/analysis/campaign-2026-09-27-v1/wide.csv \\
      --output   results/ifs-2027/analysis/plots-2026-09-27-v1 \\
      --wide     results/ifs-2027/analysis/campaign-2026-09-27-v1/wide.csv \
      --plot degenerate_share n_curve criterion_heatmap tau_sweep

    # Preview which plots would render without importing matplotlib.
    ./.venv/bin/python scripts/plot_figures.py --campaign ... --output ... \\
      --plot degenerate_share --dry-run

Plot sizes default to 6×4 inch, 150 dpi. Override with ``--figsize`` and
``--dpi``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


PLOT_SLUGS = ("degenerate_share", "n_curve", "criterion_heatmap",
              "tau_sweep", "family_boxplot")


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, default=None,
                        help="Campaign aggregator output dir (holds cell_summary.csv, variability.csv, correlations.csv)")
    parser.add_argument("--tau", type=Path, default=None,
                        help="τ-sweep output dir (holds tau_sweep.csv)")
    parser.add_argument("--family", type=Path, default=None,
                        help="Statistical-family output dir (holds family_results.csv)")
    parser.add_argument("--wide", type=Path, default=None,
                        help="wide.csv (needed for family_boxplot)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh subdirectory under results/ifs-2027/analysis/")
    parser.add_argument("--plot", nargs="+", choices=PLOT_SLUGS,
                        default=list(PLOT_SLUGS))
    parser.add_argument("--figsize", nargs=2, type=float, default=[6.0, 4.0])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be plotted; do not import matplotlib.")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="Exploratory only: write labelled no-data placeholders")
    return parser.parse_args()


def _load_csv(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_cell_summary(campaign_dir: Path | None) -> list[dict]:
    return _load_csv(campaign_dir / "cell_summary.csv" if campaign_dir else None)


def _load_variability(campaign_dir: Path | None) -> list[dict]:
    return _load_csv(campaign_dir / "variability.csv" if campaign_dir else None)


def _load_correlations(campaign_dir: Path | None) -> list[dict]:
    return _load_csv(campaign_dir / "correlations.csv" if campaign_dir else None)


def _load_tau_sweep(tau_dir: Path | None) -> list[dict]:
    return _load_csv(tau_dir / "tau_sweep.csv" if tau_dir else None)


def _load_family(family_dir: Path | None) -> list[dict]:
    return _load_csv(family_dir / "family_results.csv" if family_dir else None)


def _write_placeholder(output: Path, slug: str, reason: str) -> None:
    (output / f"{slug}__no_data.txt").write_text(
        f"Plot {slug!r} skipped: {reason}\n", encoding="utf-8")


def _missing_data(args, output: Path, slug: str, reason: str) -> str:
    if not args.allow_placeholder:
        raise ValueError(f"{slug}: {reason}")
    if not args.dry_run:
        _write_placeholder(output, slug, reason)
    return f"{slug}: placeholder ({reason})"


# ─────────────────────────────────────────────────────────────────────────────
# Per-plot data preparation (pure — testable, no matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_degenerate_share(variability_rows: list[dict]) -> list[tuple[str, float, int]]:
    """→ ``[(cell_label, degenerate_share, n_ok), ...]`` sorted descending."""
    out: list[tuple[str, float, int]] = []
    for row in variability_rows:
        share = row.get("degenerate_share")
        if share in (None, ""):
            continue
        try:
            share = float(share)
        except ValueError:
            continue
        label = "|".join(str(row.get(k, ""))
                          for k in ("corpus", "model_id", "grounding", "N", "arm"))
        try:
            n_ok = int(row.get("n_ok") or 0)
        except ValueError:
            n_ok = 0
        out.append((label, share, n_ok))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def prepare_n_curve(cell_rows: list[dict],
                     metric: str = "semantic__semantic_f1_total__mean"
                     ) -> dict[tuple[str, str, str], list[tuple[int, float, int]]]:
    """→ ``{(corpus, model_id, grounding): [(N, mean, n_ok), ...]}``."""
    from collections import defaultdict
    out: dict[tuple[str, str, str], list[tuple[int, float, int]]] = defaultdict(list)
    valid_arms = {"guided_baseline", "guided_headline", "guided_curve", "astra_cross_corpus"}
    seen_points: set[tuple[tuple[str, str, str], int]] = set()
    for row in cell_rows:
        if row.get("arm") not in valid_arms or str(row.get("ordering_id")) != "primary":
            continue
        if str(row.get("metamodel_block")).lower() not in ("true", "1"):
            continue
        try:
            N = int(row["N"])
        except (KeyError, TypeError, ValueError):
            continue
        v = row.get(metric)
        if v in (None, ""):
            continue
        try:
            v = float(v)
        except ValueError:
            continue
        try:
            n_ok = int(row.get(metric.replace("__mean", "__n_ok")) or 0)
        except (TypeError, ValueError):
            n_ok = 0
        key = (str(row.get("corpus", "")), str(row.get("model_id", "")),
                str(row.get("grounding", "")))
        point_key = (key, N)
        if point_key in seen_points:
            raise ValueError(f"Multiple guided primary-order cells at {key} N={N}; refuse to mix settings")
        seen_points.add(point_key)
        out[key].append((N, v, n_ok))
    for key in out:
        out[key].sort(key=lambda t: t[0])
    return dict(out)


def prepare_criterion_heatmap(correlation_rows: list[dict]) -> tuple[list[str], list[list[float | None]]]:
    """→ ``(criterion_names, symmetric_matrix)`` for Spearman ρ."""
    names: list[str] = []
    matrix: dict[tuple[str, str], float | None] = {}
    for row in correlation_rows:
        a, b = row.get("criterion_a"), row.get("criterion_b")
        if a is None or b is None:
            continue
        rho = row.get("spearman_rho")
        try:
            rho = float(rho) if rho not in (None, "") else None
        except ValueError:
            rho = None
        matrix[(a, b)] = rho
        matrix[(b, a)] = rho
        for name in (a, b):
            if name not in names:
                names.append(name)
    grid: list[list[float | None]] = [
        [1.0 if r == c else matrix.get((r, c)) for c in names] for r in names
    ]
    return names, grid


def prepare_tau_sweep(tau_rows: list[dict], wide_rows: list[dict]
                      ) -> dict[tuple[str, str, str, int], list[tuple[float, float, int]]]:
    """Primary-policy mean F1 per guided cell and τ, retaining sample sizes."""
    from collections import defaultdict
    from statistics import fmean
    eligible = {}
    guided = {"guided_baseline", "guided_headline", "guided_curve", "astra_cross_corpus"}
    for row in wide_rows:
        if (row.get("arm") in guided and str(row.get("ordering_id")) == "primary"
                and str(row.get("metamodel_block")).lower() in ("true", "1")
                and str(row.get("primary_semantic_eligible")).lower() == "true"):
            eligible[str(row["run_id"])] = row
    grouped: dict[tuple[tuple[str, str, str, int], float], list[float]] = defaultdict(list)
    for row in tau_rows:
        if row.get("matching_policy") != "independent_max":
            continue
        run = eligible.get(str(row.get("run_id")))
        if run is None:
            continue
        try:
            tau = float(row["tau"])
            f1 = float(row["f1_total"])
            N = int(run["N"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(run["corpus"]), str(run["model_id"]), str(run["grounding"]), N)
        grouped[(key, tau)].append(f1)
    out: dict[tuple[str, str, str, int], list[tuple[float, float, int]]] = defaultdict(list)
    for (key, tau), scores in grouped.items():
        out[key].append((tau, fmean(scores), len(scores)))
    for key in out:
        out[key].sort(key=lambda point: point[0])
    return dict(out)


def prepare_family_boxplot(family_rows: list[dict], wide_rows: list[dict]) -> list[tuple[str, list[float]]]:
    """Reconstruct the declared comparison arms and verify recorded sample sizes."""
    groups: list[tuple[str, list[float]]] = []
    for comp in family_rows:
        column = comp.get("metric")
        if not column:
            raise ValueError("Family result has no metric column")
        population = comp.get("semantic_population")
        for suffix in ("a", "b"):
            raw_filter = comp.get(f"arm_{suffix}_filter")
            if not raw_filter:
                raise ValueError(f"Missing declared arm_{suffix} filter for {comp.get('comparison')}")
            filt = json.loads(raw_filter) if isinstance(raw_filter, str) else raw_filter
            if not isinstance(filt, dict):
                raise ValueError("Arm filter must be a JSON object")
            values = []
            for row in wide_rows:
                if any(str(row.get(k)) != str(v) for k, v in filt.items()):
                    continue
                if population == "completed_extractable_final" and str(row.get("primary_semantic_eligible")).lower() != "true":
                    continue
                if population == "strict_admissible_final" and str(row.get("strict_admissible")).lower() != "true":
                    continue
                if str(row.get(f"{column}__status")) != "ok":
                    continue
                try:
                    values.append(float(row[column]))
                except (TypeError, ValueError, KeyError):
                    continue
            expected = comp.get(f"n_{suffix}")
            if expected not in (None, "") and len(values) != int(expected):
                raise ValueError(f"Arm {suffix} of {comp.get('comparison')} has {len(values)} values; results file says {expected}")
            groups.append((f"{comp.get('comparison', '?')}:{suffix}", values))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib rendering (imported lazily so --dry-run works without it)
# ─────────────────────────────────────────────────────────────────────────────

def _render_bar(labels, values, *, title, xlabel, ylabel, path, figsize, dpi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(path) + ".pdf", dpi=dpi)
    fig.savefig(str(path) + ".png", dpi=dpi)
    plt.close(fig)


def _render_lines(series, *, title, xlabel, ylabel, path, figsize, dpi, counts=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    for series_index, (label, xs, ys) in enumerate(series):
        ax.plot(xs, ys, marker="o", label=label)
        if counts is not None:
            for x, y, n in zip(xs, ys, counts[series_index]):
                ax.annotate(f"n={n}", (x, y), xytext=(3, 5),
                            textcoords="offset points", fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(str(path) + ".pdf", dpi=dpi)
    fig.savefig(str(path) + ".png", dpi=dpi)
    plt.close(fig)


def _render_heatmap(names, grid, *, title, path, figsize, dpi):
    import matplotlib.pyplot as plt
    import numpy as np
    arr = np.array([[(v if v is not None else float("nan")) for v in row] for row in grid])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(str(path) + ".pdf", dpi=dpi)
    fig.savefig(str(path) + ".png", dpi=dpi)
    plt.close(fig)


def _render_boxplot(groups, *, title, xlabel, ylabel, path, figsize, dpi):
    import matplotlib.pyplot as plt
    labels, samples = [], []
    for name, values in groups:
        if not values:
            continue
        labels.append(name)
        samples.append(values)
    fig, ax = plt.subplots(figsize=figsize)
    if samples:
        ax.boxplot(samples, labels=labels, showmeans=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(str(path) + ".pdf", dpi=dpi)
    fig.savefig(str(path) + ".png", dpi=dpi)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch(slug: str, args, output: Path) -> str:
    """Return a one-line status string for the logs."""
    figsize = tuple(args.figsize)
    if slug == "degenerate_share":
        rows = prepare_degenerate_share(_load_variability(args.campaign))
        if not rows:
            return _missing_data(args, output, slug, "variability.csv absent or empty")
        if args.dry_run:
            return f"{slug}: {len(rows)} bars would be plotted"
        _render_bar([r[0] for r in rows], [r[1] for r in rows],
                    title="Degenerate share per cell",
                    xlabel="Degenerate share (fraction of ok runs)",
                    ylabel="Cell (corpus|model|grounding|N|arm)",
                    path=output / slug, figsize=figsize, dpi=args.dpi)
        return f"{slug}: {len(rows)} bars"

    if slug == "n_curve":
        cell_rows = _load_cell_summary(args.campaign)
        series_by_key = prepare_n_curve(cell_rows)
        if not series_by_key:
            return _missing_data(args, output, slug, "cell_summary.csv absent or no eligible F1 cell")
        series = [(f"{c}|{m}|{g}", [n for n, _, _ in pts], [v for _, v, _ in pts])
                  for (c, m, g), pts in sorted(series_by_key.items())]
        if args.dry_run:
            return f"{slug}: {len(series)} lines would be plotted"
        _render_lines(series, title="F1 vs N per (corpus, model, grounding)",
                      xlabel="N", ylabel="mean F1_total",
                      path=output / slug, figsize=figsize, dpi=args.dpi,
                      counts=[[n for _, _, n in pts]
                              for _, pts in sorted(series_by_key.items())])
        return f"{slug}: {len(series)} lines"

    if slug == "criterion_heatmap":
        names, grid = prepare_criterion_heatmap(_load_correlations(args.campaign))
        if not names:
            return _missing_data(args, output, slug, "correlations.csv absent or empty")
        if args.dry_run:
            return f"{slug}: {len(names)}×{len(names)} heatmap would be plotted"
        _render_heatmap(names, grid,
                        title="Spearman ρ between criteria",
                        path=output / slug, figsize=figsize, dpi=args.dpi)
        return f"{slug}: {len(names)}×{len(names)} heatmap"

    if slug == "tau_sweep":
        rows = _load_tau_sweep(args.tau)
        if args.wide is None or not args.wide.is_file():
            return _missing_data(args, output, slug, "--wide absent")
        by_cell = prepare_tau_sweep(rows, _load_csv(args.wide))
        if not by_cell:
            return _missing_data(args, output, slug, "tau_sweep.csv absent or no eligible guided cells")
        series = [("|".join(map(str, key)), [t for t, _, _ in pts], [f for _, f, _ in pts])
                  for key, pts in sorted(by_cell.items())]
        if args.dry_run:
            return f"{slug}: {len(series)} guided cells would be plotted"
        _render_lines(series, title="Mean F1_total vs τ per guided cell",
                      xlabel="τ", ylabel="Mean F1_total",
                      path=output / slug, figsize=figsize, dpi=args.dpi,
                      counts=[[n for _, _, n in pts] for _, pts in sorted(by_cell.items())])
        return f"{slug}: {len(series)} lines"

    if slug == "family_boxplot":
        if args.wide is None or not args.wide.is_file():
            return _missing_data(args, output, slug, "--wide absent")
        family_rows = _load_family(args.family)
        wide_rows = _load_csv(args.wide)
        if not family_rows or not wide_rows:
            return _missing_data(args, output, slug, "family_results.csv or wide.csv absent or empty")
        groups = prepare_family_boxplot(family_rows, wide_rows)
        if not any(values for _, values in groups):
            return _missing_data(args, output, slug, "all declared comparison arms have zero observed scores")
        if args.dry_run:
            return f"{slug}: {len(groups)} box positions would be plotted"
        _render_boxplot(groups,
                        title="Headline arm distributions",
                        xlabel="Comparison", ylabel="Metric value",
                        path=output / slug, figsize=figsize, dpi=args.dpi)
        return f"{slug}: {len(groups)} boxes"

    raise ValueError(f"unknown plot slug: {slug}")


def main() -> int:
    args = _cli()
    output = args.output.resolve()
    allowed = (REPO / "results/ifs-2027/analysis").resolve()
    if not args.dry_run and (output.exists() or output == allowed
                              or not output.is_relative_to(allowed)):
        raise SystemExit(f"Choose a new subdirectory under {allowed.relative_to(REPO)}")

    required = {
        "degenerate_share": (args.campaign / "variability.csv" if args.campaign else None,),
        "n_curve": (args.campaign / "cell_summary.csv" if args.campaign else None,),
        "criterion_heatmap": (args.campaign / "correlations.csv" if args.campaign else None,),
        "tau_sweep": (args.tau / "tau_sweep.csv" if args.tau else None, args.wide),
        "family_boxplot": (args.family / "family_results.csv" if args.family else None, args.wide),
    }
    if not args.allow_placeholder:
        missing = [f"{slug}: {path or '<argument absent>'}" for slug in args.plot
                   for path in required[slug] if path is None or not path.is_file()]
        if missing:
            raise SystemExit("Required plotting input absent:\n" + "\n".join(missing))

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)

    print(f"Output:   {output}")
    print(f"Plots:    {args.plot}")
    if args.dry_run:
        print("(dry-run — matplotlib will not be imported; no files written)")

    lines: list[str] = []
    for slug in args.plot:
        line = _dispatch(slug, args, output)
        print(f"  · {line}")
        lines.append(line)

    if not args.dry_run:
        (output / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
