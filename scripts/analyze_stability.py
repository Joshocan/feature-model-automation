#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from fame.evaluation import compute_stability


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute stability metrics over a set of FM XML files.")
    ap.add_argument("--fm", nargs="+", required=True, help="List or glob of FM XML paths.")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model for cosine metrics.")
    ap.add_argument("--include-constraints", action="store_true", help="Include constraint cosine stability (requires embeddings).")
    ap.add_argument("--out", default="", help="Optional output JSON path; stdout if omitted.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Expand globs
    fm_paths: List[Path] = []
    for spec in args.fm:
        paths = list(Path().glob(spec))
        if paths:
            fm_paths.extend(paths)
        else:
            fm_paths.append(Path(spec))

    metrics = compute_stability(
        fm_paths,
        embed_model=args.embed_model,
        include_constraints=args.include_constraints,
    )

    payload = metrics.__dict__
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote stability metrics to {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
