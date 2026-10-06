#!/usr/bin/env python3
"""cross-run aggregation, variability, criterion divergence.

Reads long-form metrics.csv files from prior phase output directories
(``--structural`` / ``--semantic`` / ``--provenance``), joins them into a
per-run wide table, then rolls that up into per-cell summaries, variability
report and criterion-divergence rank table.

Everything is written into a fresh subdirectory under
``results/ifs-2027/analysis``:

  * ``wide.csv``           — one row per run, one column per (source, metric).
  * ``cell_summary.csv``   — one row per cell (corpus × model × N × …) with
                             mean / median / SD / n_ok per metric.
  * ``variability.csv``    — per-cell degenerate share + SD of shape metrics
                             + mean/SD of outcome columns.
  * ``rankings.json``      — ordered lists of cells per criterion + scores.
  * ``correlations.csv``   — Spearman ρ and Kendall τ between criterion pairs.
  * ``summary.json``       — hashes, source paths, status histograms.

Only the ``ok`` cells feed mean/median/SD; ``ineligible`` /
``missing_artifact`` / ``not_applicable`` are counted separately so the
denominator story is preserved (never silently zero-filled).

Usage
-----

    ./.venv/bin/python scripts/aggregate_campaign.py \\
      --structural results/ifs-2027/analysis/structure-2026-09-26-v1 \\
      --semantic   results/ifs-2027/analysis/semantic-2026-09-26-v1 \\
      --provenance results/ifs-2027/analysis/provenance-2026-09-27-v1 \\
      --output     results/ifs-2027/analysis/campaign-2026-09-27-v1
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.evaluation.campaign_aggregate import (                # noqa: E402
    add_evaluation_populations,
    DEFAULT_CELL_KEYS,
    infer_metric_columns,
    join_by_run,
    load_long_metrics,
    summarise_campaign,
    variability_report,
    write_wide_csv,
)
from fame.evaluation.criterion_divergence import (              # noqa: E402
    DEFAULT_CRITERIA,
    build_rankings,
    pairwise_correlations,
)
from fame.evaluation.inventory import sha256                    # noqa: E402


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structural", type=Path, required=True,
                        help="Structural evaluator output dir (contains metrics.csv)")
    parser.add_argument("--semantic", type=Path, default=None,
                        help="Semantic evaluator output dir (optional if only structural is ready)")
    parser.add_argument("--provenance", type=Path, default=None,
                        help="Provenance evaluator output dir (optional)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh subdirectory under results/ifs-2027/analysis/")
    parser.add_argument("--cell-key", action="append", default=None,
                        help="Override cell key (repeatable). Default: %s" % ", ".join(DEFAULT_CELL_KEYS))
    return parser.parse_args()


def _row_to_csv_scalar(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def _metrics_csv_path(root: Path) -> Path:
    """Resolve CLI input paths before recording them relative to the repo."""
    return (root / "metrics.csv").resolve()


def _write_rows(rows, path: Path) -> None:
    if not rows:
        return
    fields = list({k for row in rows for k in row})
    fields.sort()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _row_to_csv_scalar(row.get(k)) for k in fields})


def main() -> int:
    args = _cli()
    output = args.output.resolve()
    allowed = (REPO / "results/ifs-2027/analysis").resolve()
    if output.exists() or output == allowed or not output.is_relative_to(allowed):
        raise SystemExit(f"Choose a new subdirectory under {allowed.relative_to(REPO)}")

    cell_keys = tuple(args.cell_key) if args.cell_key else DEFAULT_CELL_KEYS

    sources: dict[str, list[dict]] = {}
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for tag, root in [("structural", args.structural),
                       ("semantic",   args.semantic),
                       ("provenance", args.provenance)]:
        if root is None:
            continue
        csv_path = _metrics_csv_path(root)
        if not csv_path.is_relative_to(REPO):
            raise SystemExit(f"{tag} metrics.csv must be inside {REPO}: {csv_path}")
        if not csv_path.is_file():
            raise SystemExit(f"{tag} metrics.csv not found under {root}")
        sources[tag] = load_long_metrics(csv_path)
        source_paths[tag] = str(csv_path.relative_to(REPO))
        source_hashes[tag] = sha256(csv_path)

    print(f"Output:      {output.relative_to(REPO)}")
    print(f"Cell keys:   {list(cell_keys)}")
    for tag, path in source_paths.items():
        print(f"  {tag:11s} {path}  ({len(sources[tag])} rows)")

    wide_rows = add_evaluation_populations(join_by_run(sources, cell_keys=cell_keys))
    print(f"Wide rows:   {len(wide_rows)}")

    metric_columns = infer_metric_columns(wide_rows)
    cell_summary = summarise_campaign(wide_rows, metric_columns, cell_keys=cell_keys)
    var_report = variability_report(wide_rows, cell_keys=cell_keys)

    criteria = {name: col for name, col in DEFAULT_CRITERIA.items()
                if any(col in row for row in cell_summary)}
    if criteria:
        rankings = build_rankings(cell_summary, cell_id_keys=cell_keys, criteria=criteria)
        correlations = pairwise_correlations(cell_summary, criteria)
    else:
        rankings = []
        correlations = []

    output.mkdir(parents=True, exist_ok=False)
    write_wide_csv(wide_rows, output / "wide.csv")
    _write_rows(cell_summary, output / "cell_summary.csv")
    _write_rows(var_report,    output / "variability.csv")

    rankings_out = [
        dict(criterion=r.criterion,
             cell_ids=[list(cid) for cid in r.cell_ids],
             scores=r.scores, ranks=r.ranks)
        for r in rankings
    ]
    (output / "rankings.json").write_text(
        json.dumps(dict(cell_keys=list(cell_keys), rankings=rankings_out), indent=2, default=str) + "\n")

    correlation_rows = [dict(criterion_a=c.criterion_a, criterion_b=c.criterion_b,
                              n=c.n, n_excluded=c.n_excluded, status=c.status,
                              spearman_rho=c.spearman_rho, kendall_tau=c.kendall_tau)
                        for c in correlations]
    _write_rows(correlation_rows, output / "correlations.csv")

    versions = {}
    for pkg in ("numpy", "lxml"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None

    status_hist = Counter()
    for row in wide_rows:
        for k, v in row.items():
            if k.endswith("__status") and v is not None:
                status_hist[v] += 1

    result = dict(
        n_runs=len(wide_rows),
        n_cells=len(cell_summary),
        cell_keys=list(cell_keys),
        sources=source_paths,
        source_sha256=source_hashes,
        n_metric_columns=len(metric_columns),
        criteria_used=list(criteria),
        rank_correlations=correlation_rows,
        status_histogram=dict(status_hist),
        implementation_sha256={p: sha256(REPO / p) for p in (
            "fame/evaluation/campaign_aggregate.py",
            "fame/evaluation/criterion_divergence.py",
            "scripts/aggregate_campaign.py",
        )},
        versions=versions,
        publication_ready=False,
        unresolved=["D02 final approval of strict-admissibility sensitivity gate",
                    "D04 comparison-family definitions"],
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("implementation_sha256", "versions", "source_sha256")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
