#!/usr/bin/env python3
"""Phase 4.6 — build Chroma index per corpus from chunks.jsonl.

Reads ``config/experiment.yaml`` to locate chunks and the Chroma root.
The collection is fully reset on each run for reproducibility.

Usage:
  python scripts/build_index.py                         # both corpora
  python scripts/build_index.py --corpus federation
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.vectorization import build_index, OllamaEmbedder  # noqa: E402


def _load_cfg() -> dict:
    return yaml.safe_load((REPO / "config/experiment.yaml").read_text())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["federation", "repair", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chroma-root", default=None,
                   help="Override chroma root; default is data/chroma (per IFS target shape).")
    return p.parse_args()


def main() -> int:
    args = _cli()
    cfg = _load_cfg()
    chroma_root = Path(args.chroma_root) if args.chroma_root else (REPO / "data/chroma")
    embedder = OllamaEmbedder()

    corpora = ["federation", "repair"] if args.corpus == "both" else [args.corpus]

    for corpus in corpora:
        chunks_jsonl = REPO / f"data/processed/{corpus}/chunks.jsonl"
        print(f"\n===== {corpus}: index → {chroma_root / corpus} =====")
        t0 = time.time()

        def _progress(done: int, total: int) -> None:
            print(f"  [{corpus}] {done}/{total} embedded", end="\r", flush=True)

        report = build_index(
            corpus=corpus,
            chunks_jsonl=chunks_jsonl,
            chroma_root=chroma_root,
            embedder=embedder,
            batch_size=args.batch_size,
            on_progress=_progress,
        )
        dt = time.time() - t0
        print()   # newline after progress line
        print(f"  read:     {report.chunks_read}")
        print(f"  upserted: {report.chunks_upserted}")
        print(f"  failed:   {report.chunks_failed}")
        print(f"  time:     {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
