#!/usr/bin/env python3
"""Phase 4a — build canonical chunks.jsonl per corpus.

Reads chunking policy from ``config/experiment.yaml`` and emits:
  data/processed/federation/chunks.jsonl
  data/processed/repair/chunks.jsonl
  data/processed/chunks_stats.json

Usage:
  python scripts/build_chunks.py                # both corpora
  python scripts/build_chunks.py --corpus federation
  python scripts/build_chunks.py --corpus repair --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.ingestion import (               # noqa: E402
    ChunkingConfig,
    build_corpus_chunks,
    summarise_reports,
    DocReport,
)


def _load_config() -> dict:
    return yaml.safe_load((REPO / "config/experiment.yaml").read_text())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["federation", "repair", "both"], default="both")
    p.add_argument("--limit", type=int, default=None,
                   help="For debugging: process only the first N documents of each corpus.")
    return p.parse_args()


def _make_progress_callbacks(corpus: str):
    idx = {"n": 0}
    def on_start(doc_id: str) -> None:
        idx["n"] += 1
        print(f"  [{corpus}] {idx['n']:>3} {doc_id} ...", end="", flush=True)
    def on_done(r: DocReport) -> None:
        if r.error:
            print(f" ERROR ({r.seconds:.1f}s): {r.error}")
        else:
            print(f" {r.n_chunks:>3} chunks | {r.text_chars:>6} chars | {r.seconds:.1f}s")
    return on_start, on_done


def main() -> int:
    args = _cli()
    cfg_all = _load_config()
    chunk_cfg = cfg_all["chunking"]

    config = ChunkingConfig(
        max_chunk_chars=int(chunk_cfg["max_chunk_chars"]),
        overlap_chars=int(round(chunk_cfg["max_chunk_chars"] * float(chunk_cfg["overlap_ratio"]))),
        preprocessing_version=chunk_cfg["preprocessing_version"],
    )

    print(f"Chunking policy: max={config.max_chunk_chars} chars, "
          f"overlap={config.overlap_chars} chars, "
          f"version={config.preprocessing_version}")
    print()

    stats: dict = {
        "preprocessing_version": config.preprocessing_version,
        "chunk_size_chars": config.max_chunk_chars,
        "overlap_chars": config.overlap_chars,
    }

    corpora = ["federation", "repair"] if args.corpus == "both" else [args.corpus]
    for corpus in corpora:
        c = cfg_all["corpora"][corpus]
        print(f"===== {corpus} =====")
        on_start, on_done = _make_progress_callbacks(corpus)

        # honour --limit by trimming the manifest in place isn't ideal; instead,
        # patch _read_doc_ids at call time via monkey.
        if args.limit is not None:
            from fame.ingestion import pipeline as _p
            original = _p._read_doc_ids_from_manifest
            _p._read_doc_ids_from_manifest = lambda m, _o=original: _o(m)[:args.limit]

        reports = build_corpus_chunks(
            corpus=corpus,
            manifest_csv=REPO / c["manifest"],
            raw_dir=REPO / c["raw_dir"],
            out_jsonl=REPO / f"data/processed/{corpus}/chunks.jsonl",
            config=config,
            on_doc_start=on_start,
            on_doc_done=on_done,
        )

        if args.limit is not None:
            _p._read_doc_ids_from_manifest = original  # noqa

        summary = summarise_reports(corpus, reports)
        stats[corpus] = summary
        print()
        print(f"  {corpus}: {summary['docs_ok']}/{summary['docs_total']} ok "
              f"| {summary['total_chunks']} chunks total "
              f"| chunks/doc: min={summary['min']}, median={summary['median']}, "
              f"mean={summary['mean']}, max={summary['max']} "
              f"| {summary['seconds_total']:.1f}s")
        print()

    stats_path = REPO / "data/processed/chunks_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
