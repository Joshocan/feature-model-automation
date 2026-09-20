#!/usr/bin/env python
"""Extract feature list from a FeatureIDE XML and emit JSON.

Usage:
  PYTHONPATH=$(pwd) .venv/bin/python scripts/extract_features.py \
    --xml results/rag/ss-rgfm/fm/some_model.xml \
    --out results/rag/ss-rgfm/fm/some_model.features.json

If --out is omitted, prints JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fame.evaluation.feature_list import extract_feature_list


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract feature list to JSON")
    ap.add_argument("--xml", required=True, help="FeatureIDE XML input")
    ap.add_argument("--out", help="Output JSON path (default: stdout)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records = extract_feature_list(args.xml)
    data = [r.to_dict() for r in records]

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
