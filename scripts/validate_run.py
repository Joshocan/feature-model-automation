#!/usr/bin/env python3
"""CLI — validate a run directory against the logging contract.

Usage:
  python scripts/validate_run.py results/<campaign>/<corpus>/<config>/<run_id>/
  python scripts/validate_run.py results/<...> --json > report.json

Exits 0 only when the run is execution-complete and admissible, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.validation import validate_run  # noqa: E402


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", type=Path)
    p.add_argument("--json", action="store_true", help="emit JSON only, no human summary")
    return p.parse_args()


def main() -> int:
    args = _cli()
    report = validate_run(args.run_root, repo_root=REPO)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return 0 if report.admissible else 1

    print(f"===== validate: {report.run_root} =====")
    print(f"  run_id:     {report.run_id}")
    print(f"  execution_complete: {report.execution_complete}")
    print(f"  xsd_valid:         {report.xsd_valid}")
    print(f"  admissible:        {report.admissible}")
    print(f"  errors:     {report.n_errors}")
    print(f"  warnings:   {report.n_warnings}")
    for f in report.findings:
        print(f"  [{f.severity.value:7s}] {f.check}: {f.detail}")
    return 0 if report.admissible else 1


if __name__ == "__main__":
    raise SystemExit(main())
