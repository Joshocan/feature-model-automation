#!/usr/bin/env python3
"""
Run the FAME ingestion + preparation pipeline.

Default behavior remains repo-global:
  - reads from data/raw
  - writes chunks under data/processed/algorithm_1/chunks

Dataset-aware usage is also supported:
  - --dataset federation -> data/raw/federation -> data/processed/federation/chunks
  - --dataset repair     -> data/raw/repair     -> data/processed/repair/chunks
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fame.ingestion.pipeline import ingest_and_prepare
from fame.nonrag.cli_utils import normalize_dataset_name
from fame.utils.runtime import workspace


DATASET_PROCESSED_SUBDIR = {
    "federation": "federation",
    "repair": "repair",
}


def _resolve_paths(paths, dataset: str | None, raw_dir: str | None, out_dir: str | None) -> tuple[Path, Path, str | None]:
    dataset_name = normalize_dataset_name(dataset) if dataset else None
    if raw_dir:
        resolved_raw = Path(raw_dir).expanduser().resolve()
    elif dataset_name:
        resolved_raw = (paths.base_dir / "data" / "raw" / dataset_name).resolve()
    else:
        resolved_raw = paths.raw_data

    if out_dir:
        resolved_out = Path(out_dir).expanduser().resolve()
    elif dataset_name:
        subdir = DATASET_PROCESSED_SUBDIR.get(dataset_name, dataset_name)
        resolved_out = (paths.base_dir / "data" / "processed" / subdir / "chunks").resolve()
    else:
        resolved_out = (paths.processed_data / "chunks").resolve()

    return resolved_raw, resolved_out, dataset_name


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run FAME ingestion/chunking")
    ap.add_argument("--dataset", choices=("federation", "repair", "fed", "federated", "model_repair"), default="", help="Dataset preset for raw/chunk directories")
    ap.add_argument("--raw-dir", default="", help="Explicit input directory containing source PDFs")
    ap.add_argument("--out-dir", default="", help="Explicit output directory for *.chunks.json")
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    ws = workspace("preprocess", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths
    raw_dir, out_dir, dataset_name = _resolve_paths(
        paths,
        dataset=args.dataset or None,
        raw_dir=args.raw_dir or None,
        out_dir=args.out_dir or None,
    )

    print("=== FAME Ingestion Stage ===")
    print(f"Base dir       : {paths.base_dir}")
    print(f"Dataset        : {dataset_name or '(default)'}")
    print(f"Raw data dir   : {raw_dir}")
    print(f"Chunks out dir : {out_dir}")
    print("============================")

    result = ingest_and_prepare(raw_dir=raw_dir, out_dir=out_dir)

    print("\n=== Ingestion Result ===")
    print(f"Processed files: {len(result['processed'])}")
    print(f"Skipped files  : {len(result['skipped'])}")

    if result["processed"]:
        print("\nSample output:")
        print(f"  {result['processed'][0]}")

    if result["skipped"]:
        print("\nSkipped inputs:")
        for s in result["skipped"]:
            print(f"  - {s}")

    print("\nSUCCESS: Ingestion stage completed successfully.")


if __name__ == "__main__":
    main()
