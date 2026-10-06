#!/usr/bin/env python3
"""Read-only reconciliation of one IFS evaluation snapshot; no model calls.

Each stage's long metric table and the campaign wide table must account
for every planned inventory run. This checks denominators, not scientific
validity or the still-open evaluation decisions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Required table is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(inventory: Path, structural: Path, semantic: Path,
           provenance: Path, aggregate: Path) -> dict:
    runs_file = inventory / "runs.json"
    if not runs_file.is_file():
        raise ValueError(f"Required inventory is missing: {runs_file}")
    planned = json.loads(runs_file.read_text(encoding="utf-8"))
    ids = [str(run["run_id"]) for run in planned]
    if len(ids) != len(set(ids)):
        raise ValueError("Inventory contains duplicate run IDs")
    expected = set(ids)
    runs_hash = _sha256(runs_file)
    artifacts_hash = _sha256(inventory / "artifacts.json")
    stages = {}
    for name, directory in (("structural", structural), ("semantic", semantic),
                            ("provenance", provenance)):
        rows = _csv_rows(directory / "metrics.csv")
        actual = {row.get("run_id") for row in rows}
        missing, extra = expected - actual, actual - expected
        if missing or extra:
            raise ValueError(f"{name}: run-ID mismatch, missing={len(missing)}, extra={len(extra)}")
        duplicate_pairs = sum(count - 1 for count in
            Counter((row.get("run_id"), row.get("metric")) for row in rows).values()
            if count > 1)
        if duplicate_pairs:
            raise ValueError(f"{name}: {duplicate_pairs} duplicate run/metric pairs")
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"{name}: missing summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("inventory_sha256") != runs_hash or
                summary.get("artifacts_sha256") != artifacts_hash):
            raise ValueError(f"{name}: input hashes do not match the selected inventory")
        stages[name] = {"rows": len(rows), "runs": len(actual),
                        "statuses": dict(Counter(row.get("status") for row in rows))}
    wide = _csv_rows(aggregate / "wide.csv")
    wide_ids = [row.get("run_id") for row in wide]
    if len(wide_ids) != len(set(wide_ids)) or set(wide_ids) != expected:
        raise ValueError("Aggregate wide.csv does not contain exactly one row per planned run")
    for field in ("primary_semantic_eligible", "strict_admissible"):
        if field not in wide[0]:
            raise ValueError(f"Aggregate is missing population flag {field}")
    aggregate_summary_path = aggregate / "summary.json"
    if not aggregate_summary_path.is_file():
        raise ValueError("Aggregate is missing summary.json")
    aggregate_summary = json.loads(aggregate_summary_path.read_text(encoding="utf-8"))
    for name, directory in (("structural", structural), ("semantic", semantic),
                            ("provenance", provenance)):
        if aggregate_summary.get("source_sha256", {}).get(name) != _sha256(directory / "metrics.csv"):
            raise ValueError(f"Aggregate source hash for {name} differs from the selected metrics.csv")
    populations = {
        field: dict(Counter(row.get(field) or "unknown" for row in wide))
        for field in ("primary_semantic_eligible", "strict_admissible")
    }
    completed = sum(run.get("inventory_status") == "completed" for run in planned)
    return {"planned_runs": len(planned), "completed_runs": completed,
            "stages": stages, "populations": populations,
            "status": "denominators_reconciled"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ("inventory", "structural", "semantic", "provenance", "aggregate"):
        parser.add_argument(f"--{stage}", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.inventory, args.structural, args.semantic,
                        args.provenance, args.aggregate)
    except ValueError as exc:
        parser.exit(1, f"Evaluation verification failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
