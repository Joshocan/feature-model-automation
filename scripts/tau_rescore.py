#!/usr/bin/env python3
"""Re-threshold pairs.csv at each τ; emit tau_sweep.csv.

Reads the raw D15 cosine similarities written by ``evaluate_semantic.py``
and re-scores every run at each τ ∈ {0.3, 0.4, 0.5, 0.6} (or a
user-supplied list). No embedding or generation call; matching is recomputed.

Outputs per (run, τ, matching_policy):
  * n_generated, n_reference, n_matched_pairs
  * precision, recall_total, f1_total
  * matched_reference_ids (JSON list)

Downstream analysis pipes this through ``analyse_family.py`` to compare
whether the family's Holm-survivors flip under different τ.

Usage
-----

    ./.venv/bin/python scripts/tau_rescore.py \\
      --pairs   results/ifs-2027/analysis/semantic-2026-09-26-v1/pairs.csv \\
      --output  results/ifs-2027/analysis/tau-sweep-2026-09-27-v1

    # Custom τ list + one-to-one sensitivity as well
    ./.venv/bin/python scripts/tau_rescore.py \\
      --pairs   ... \\
      --output  ... \\
      --tau     0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \\
      --policy  independent_max one_to_one
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.evaluation.inventory import sha256                     # noqa: E402
from fame.evaluation.tau_rescore import rescore_run              # noqa: E402


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="pairs.csv from the semantic evaluator — raw cosine similarities per run")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh subdirectory under results/ifs-2027/analysis/")
    parser.add_argument("--tau", type=float, nargs="+",
                        default=[0.3, 0.4, 0.5, 0.6],
                        help="τ values to sweep (default from contract)")
    parser.add_argument("--policy", type=str, nargs="+",
                        default=["independent_max"],
                        choices=["independent_max", "one_to_one", "one_to_one_max_weight"],
                        help="Matching policies to score under (default: primary only)")
    return parser.parse_args()


def _iter_run_pairs(path: Path):
    """Bound memory to one run; reject malformed or interleaved input."""
    with path.open(newline="", encoding="utf-8") as fh:
        group, current, seen = [], None, set()
        for line, row in enumerate(csv.DictReader(fh), 2):
            try:
                row["similarity"] = float(row.get("similarity", 0.0) or 0.0)
                row["gen_index"] = int(row["gen_index"])
                row["ref_index"] = int(row["ref_index"])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{line}: invalid similarity pair") from exc
            rid = row['run_id']
            if not rid or not math.isfinite(row['similarity']) or not -1.00001 <= row['similarity'] <= 1.00001:
                raise ValueError(f'{path}:{line}: invalid run ID or cosine similarity')
            if rid != current:
                if rid in seen:
                    raise ValueError(f"{path}: run {rid} is not contiguous")
                if group:
                    yield group
                seen.add(rid)
                current, group = rid, []
            group.append(row)
        if group:
            yield group


def main() -> int:
    args = _cli()
    output = args.output.resolve()
    allowed = (REPO / "results/ifs-2027/analysis").resolve()
    if output.exists() or output == allowed or not output.is_relative_to(allowed):
        raise SystemExit(f"Choose a new subdirectory under {allowed.relative_to(REPO)}")

    print(f"Pairs:       {args.pairs} (streaming by run)")
    print(f"Output:      {output.relative_to(REPO)}")
    print(f"τ values:    {args.tau}")
    print(f"Policies:    {args.policy}")

    scores, n_pair_rows = [], 0
    for pairs in _iter_run_pairs(args.pairs.resolve()):
        n_pair_rows += len(pairs)
        scores.extend(rescore_run(pairs, tau=tau, matching_policy=policy)
                      for tau in args.tau for policy in args.policy)
    print(f"Score rows:  {len(scores)}")

    output.mkdir(parents=True, exist_ok=False)
    rows = [dict(
        run_id=s.run_id, tau=s.tau, matching_policy=s.matching_policy,
        n_generated=s.n_generated, n_reference=s.n_reference,
        n_generated_matched=s.n_generated_matched,
        n_reference_matched=s.n_reference_matched,
        n_matched_pairs=s.n_matched_pairs,
        precision=s.precision, recall_total=s.recall_total, f1_total=s.f1_total,
        matched_reference_ids=json.dumps(s.matched_reference_ids),
    ) for s in scores]
    fields = list(rows[0].keys()) if rows else []
    with (output / "tau_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    versions = {}
    for package in ('numpy', 'scipy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    summary = dict(
        n_pair_rows=n_pair_rows,
        n_score_rows=len(scores),
        n_runs=len({s.run_id for s in scores}),
        tau_values=list(args.tau),
        matching_policies=list(args.policy),
        matching_objectives={'one_to_one': 'maximum cardinality then maximum total cosine weight',
                             'one_to_one_max_weight': 'maximum total cosine weight, allowing fewer pairs'},
        policy_counter=dict(Counter(s.matching_policy for s in scores)),
        pairs_sha256=sha256(args.pairs.resolve()),
        implementation_sha256={p: sha256(REPO / p) for p in (
            "fame/evaluation/tau_rescore.py",
            "scripts/tau_rescore.py",
        )},
        versions=versions,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                       if k not in ("implementation_sha256", "versions")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
