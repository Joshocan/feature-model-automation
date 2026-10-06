#!/usr/bin/env python3
"""declared comparison family with Holm + Cliff's δ + bootstrap CIs.

Reads the campaign aggregator's ``wide.csv`` plus a family declaration JSON, and writes:

  * ``family_results.csv``           — one row per declared comparison
  * ``borderline_nonsurvivors.csv``  — nominally significant, not survived Holm
  * ``bootstrap_intervals.csv``      — mean + 95% CI per (arm, metric)
  * ``summary.json``                 — hashes, versions, family metadata

Every comparison is Mann–Whitney U with two-sided p, Cliff's δ effect size,
and a **minimum-n gate** (default 5 per arm). Comparisons below min-n or
with an empty arm are recorded as ``status="insufficient_n"`` /
``"empty_arm"`` — the paper table needs both denominators.

Family declaration format
-------------------------

JSON file with the following shape::

    {
      "family": "H1b_headline",
      "min_n": 5,
      "alpha": 0.05,
      "comparisons": [
        {
          "name": "H1b_repair_N10_rag_vs_nonrag",
          "arm_a": {"grounding": "rag",    "N": 10, "corpus": "repair"},
          "arm_b": {"grounding": "nonrag", "N": 10, "corpus": "repair"},
          "metric": "semantic__semantic_f1_total"
        },
        ...
      ]
    }

Arm filters are equality dicts against the wide.csv columns; a run
matches if every declared column matches. Additional filters (e.g.
``"model_id": "deepseek-v4.1"``) can be added to isolate subarms.

Usage
-----

    ./.venv/bin/python scripts/analyse_family.py \\
      --wide       results/ifs-2027/analysis/campaign-2026-09-27-v1/wide.csv \\
      --family     config/analysis/families/h1b_headline.json \\
      --output     results/ifs-2027/analysis/family-h1b-2026-09-27-v1

The output directory must be a fresh subdirectory under
``results/ifs-2027/analysis``.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.evaluation.bootstrap import mean_ci                    # noqa: E402
from fame.evaluation.campaign_aggregate import load_long_metrics # noqa: E402  # for CSV coercion
from fame.evaluation.inventory import sha256                     # noqa: E402
from fame.evaluation.statistical_family import Comparison, run_family  # noqa: E402


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wide", type=Path, required=True,
                        help="wide.csv from campaign aggregation")
    parser.add_argument("--family", type=Path, required=True,
                        help="JSON declaring the family + comparisons")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh subdirectory under results/ifs-2027/analysis/")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip per-arm bootstrap CIs (faster; still emits point means)")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def _load_wide(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _coerce(value):
    if value is None or value == "":
        return None
    s = str(value)
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        f = float(s)
        return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f
    except ValueError:
        return s


def _filter_arm(rows: list[dict], arm_filter: dict) -> list[dict]:
    def matches(row: dict) -> bool:
        for k, expected in arm_filter.items():
            observed = row.get(k)
            if observed is None:
                return False
            # Coerce both sides to string for tolerant comparison.
            if str(observed) != str(expected):
                return False
        return True
    return [r for r in rows if matches(r)]


def _write_rows(rows, path: Path) -> None:
    if not rows:
        return
    fields = list({k for row in rows for k in row})
    fields.sort()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(row.get(k)) if isinstance(row.get(k), (dict, list))
                                  else row.get(k)) for k in fields})


def main() -> int:
    args = _cli()
    output = args.output.resolve()
    allowed = (REPO / "results/ifs-2027/analysis").resolve()
    if output.exists() or output == allowed or not output.is_relative_to(allowed):
        raise SystemExit(f"Choose a new subdirectory under {allowed.relative_to(REPO)}")

    wide = _load_wide(args.wide.resolve())
    family_decl = json.loads(args.family.read_text())

    family_name = family_decl.get("family", "unnamed_family")
    min_n = int(family_decl.get("min_n", 5))
    alpha = float(family_decl.get("alpha", 0.05))
    semantic_population = family_decl.get("semantic_population", "completed_extractable_final")
    if semantic_population not in ("completed_extractable_final", "strict_admissible_final"):
        raise SystemExit("Unknown semantic population in family declaration")
    comps_raw = family_decl.get("comparisons", [])
    if not comps_raw or len({c['name'] for c in comps_raw}) != len(comps_raw):
        raise SystemExit('Family must contain uniquely named comparisons')

    print(f"Family:      {family_name}")
    print(f"min_n:       {min_n}   alpha={alpha}")
    print(f"Wide:        {args.wide}")
    print(f"Comparisons: {len(comps_raw)}")
    print(f"Output:      {output.relative_to(REPO)}")

    comparisons: list[Comparison] = []
    data_by_arm: dict[str, list[dict]] = {}
    for c in comps_raw:
        arm_a_key = f"{c['name']}::a"
        arm_b_key = f"{c['name']}::b"
        if c["metric"].startswith("semantic__"):
            gate = ("primary_semantic_eligible" if semantic_population == "completed_extractable_final"
                    else "strict_admissible")
            if not wide or gate not in wide[0]:
                raise SystemExit(f"Wide table lacks {gate}; rerun updated aggregation")
            eligible = [row for row in wide if row.get(gate) is True]
        else:
            eligible = wide
        # The same observed-value population must feed the test, the
        # bootstrap interval, and the plot. Never treat a numeric payload
        # attached to a non-ok metric as a valid observation.
        eligible = [row for row in eligible if row.get(f"{c['metric']}__status") == "ok"]
        data_by_arm[arm_a_key] = _filter_arm(eligible, c["arm_a"])
        data_by_arm[arm_b_key] = _filter_arm(eligible, c["arm_b"])
        comparisons.append(Comparison(
            name=c["name"], arm_a=arm_a_key, arm_b=arm_b_key,
            metric=c["metric"],
        ))

    report = run_family(
        family=family_name, comparisons=comparisons,
        data_by_arm=data_by_arm, min_n=min_n, alpha=alpha,
    )

    declared_by_name = {c["name"]: c for c in comps_raw}
    rows = []
    for r in report.results:
        declaration = declared_by_name[r.name]
        rows.append(dict(
            family=family_name, comparison=r.name, metric=r.metric,
            arm_a_filter=declaration["arm_a"], arm_b_filter=declaration["arm_b"],
            semantic_population=semantic_population if r.metric.startswith("semantic__") else "all_metric_eligible",
            planned_a=len(_filter_arm(wide, declaration['arm_a'])),
            planned_b=len(_filter_arm(wide, declaration['arm_b'])),
            n_a=r.n_a, n_b=r.n_b, mean_a=r.mean_a, mean_b=r.mean_b,
            p_value=r.p_value, p_adjusted=r.p_adjusted,
            cliffs_delta=r.cliffs_delta, delta_magnitude=r.delta_magnitude,
            survives_holm=r.survives_holm, borderline=r.borderline_nonsurvivor,
            status=r.status, reason=r.reason,
        ))

    output.mkdir(parents=True, exist_ok=False)
    _write_rows(rows, output / "family_results.csv")

    borderline_rows = [row for row in rows if row["borderline"]]
    _write_rows(borderline_rows, output / "borderline_nonsurvivors.csv")

    bootstrap_rows = []
    if not args.no_bootstrap:
        print("Computing bootstrap CIs ...")
        seen: set[tuple[str, str]] = set()
        for comp in comparisons:
            for arm_key, arm_label in ((comp.arm_a, "a"), (comp.arm_b, "b")):
                key = (comp.name, arm_label)
                if key in seen:
                    continue
                seen.add(key)
                values = [row.get(comp.metric) for row in data_by_arm[arm_key]
                          if isinstance(row.get(comp.metric), (int, float, bool))
                          and row.get(comp.metric) is not None]
                values = [float(v) for v in values if math.isfinite(v)]
                ci = mean_ci(values, n_resamples=args.bootstrap_resamples) if len(values) >= min_n else None
                bootstrap_rows.append(dict(
                    comparison=comp.name, arm=arm_label, metric=comp.metric,
                    n=len(values),
                    status='ok' if ci else 'descriptive_only_insufficient_n',
                    mean=(sum(values) / len(values)) if values else None,
                    sd=statistics.stdev(values) if len(values)>1 else None,
                    median=statistics.median(values) if values else None,
                    ci_lower=(ci.lower if ci else None),
                    ci_upper=(ci.upper if ci else None),
                    confidence=(ci.confidence if ci else None),
                    n_resamples=(ci.n_resamples if ci else None),
                ))
    _write_rows(bootstrap_rows, output / "bootstrap_intervals.csv")

    versions = {}
    for pkg in ("numpy", "scipy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None

    summary = dict(
        family=family_name, alpha=alpha, min_n=min_n,
        semantic_population=semantic_population,
        timing=family_decl.get('timing', 'unspecified'),
        tau_primary=family_decl.get('tau_primary'),
        multiplicity='Holm over all declared comparisons; untestable slots retained internally as p=1',
        bootstrap_seed=20260927,
        interval_method='percentile_bootstrap_95',
        n_comparisons=len(rows),
        n_tested=report.n_tested,
        n_significant_pre_holm=report.n_significant_pre,
        n_significant_post_holm=report.n_significant_post,
        n_borderline_nonsurvivors=report.n_borderline,
        wide_sha256=sha256(args.wide.resolve()),
        family_declaration_sha256=sha256(args.family.resolve()),
        implementation_sha256={p: sha256(REPO / p) for p in (
            "fame/evaluation/statistical_family.py",
            "fame/evaluation/bootstrap.py",
            "scripts/analyse_family.py",
        )},
        versions=versions,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                       if k not in ("implementation_sha256", "versions")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
