#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Dict

from fame.evaluation.constraints import extract_constraints


def count_constraints(xml_path: Path) -> Dict[str, int]:
    cons = extract_constraints(xml_path)
    counts = Counter(c.constraint_type for c in cons)
    counts["total"] = len(cons)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Count constraints in a FeatureIDE XML (requires/excludes/etc.)")
    ap.add_argument("--xml", required=True, help="FeatureIDE FM XML path")
    ap.add_argument("--out", default="", help="Optional output JSON path (default: print to stdout)")
    args = ap.parse_args()

    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    counts = count_constraints(xml_path)

    if args.out.strip():
        out_path = Path(args.out).expanduser().resolve()
        out_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
        print(f"Saved constraint counts to {out_path}")
    else:
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
