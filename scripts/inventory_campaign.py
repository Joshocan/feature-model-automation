#!/usr/bin/env python3
"""Inventory frozen campaign artifacts locally; never invokes a generation client."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.evaluation.inventory import build_inventory, read_object, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=REPO / "config/evaluation/ifs-2027-v0.1.0.json")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--path-map", type=Path, help="JSON object mapping original directory prefixes to relocated repository-relative prefixes")
    args = parser.parse_args()
    output = args.output.resolve()
    # Only write inside an analysis subtree, never raw runs, archives or source files.
    allowed = (REPO / "results/ifs-2027/analysis").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        parser.error(f"--output must be a new subdirectory of {allowed}")
    if output.exists():
        parser.error("Output already exists; choose a new inventory snapshot name")
    contract = read_object(args.contract)
    mapping = read_object(args.path_map) if args.path_map else {}
    result = build_inventory(REPO, contract, mapping)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("runs", "recovery_snapshots", "artifacts", "pilots", "issues"):
        rows = result[name]
        (output / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    result["summary"].update(specification_version=contract["specification_version"],
        contract_sha256=sha256(args.contract), created_at=datetime.now(timezone.utc).isoformat(),
        path_map=mapping, inventory_implementation_sha256=sha256(REPO / "fame/evaluation/inventory.py"))
    (output / "summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    return 1 if result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
