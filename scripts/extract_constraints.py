#!/usr/bin/env python
"""Extract constraint list from a FeatureIDE XML and emit JSON.

Usage:
  PYTHONPATH=$(pwd) .venv/bin/python scripts/extract_constraints.py \
    --xml results/rag/ss-rgfm/fm/some_model.xml \
    --out results/rag/ss-rgfm/fm/some_model.constraints.json

If --out is omitted, prints JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fame.evaluation.constraints import extract_constraints


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract constraints to JSON")
    ap.add_argument("--xml", required=True, help="FeatureIDE XML input")
    ap.add_argument("--out", help="Output JSON path (default: stdout)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records = extract_constraints(args.xml)
    data = [r.to_dict() for r in records]

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
