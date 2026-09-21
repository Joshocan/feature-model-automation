#!/usr/bin/env python3
"""Phase 4.V — retrieval validity smoke check.

For each preselected doc_id, run the 4 fixed sub-queries with the doc_id
filter set to that single doc. Print the top-10 chunks. This is qualitative:
methodology / approach chunks are the target; related-work, background, or
reference-list chunks indicate a preprocessing regression.

Preselected docs are frozen: **fed_05, rep_25, rep_50** (Phase 4.V decision).

Usage:
  python scripts/validate_retrieval.py
  python scripts/validate_retrieval.py --top-k 5
  python scripts/validate_retrieval.py --domain "model federation" --domain-repair "model repair"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.retrieval import RetrievalService  # noqa: E402

SMOKE_DOCS = [
    ("federation", "fed_05"),
    ("repair",     "rep_25"),
    ("repair",     "rep_50"),
]


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--chroma-root", default=str(REPO / "data/chroma"))
    p.add_argument("--domain-federation", default="model federation")
    p.add_argument("--domain-repair",     default="model repair")
    p.add_argument("--out", default=str(REPO / "results/smoke/retrieval_validity.md"))
    return p.parse_args()


def _preview(text: str, n: int = 240) -> str:
    t = text.replace("\n", " ").strip()
    return t if len(t) <= n else t[:n].rstrip() + "…"


def main() -> int:
    args = _cli()
    domains = {"federation": args.domain_federation, "repair": args.domain_repair}
    services: dict[str, RetrievalService] = {}
    lines: list[str] = []
    lines.append(f"# Retrieval validity smoke — Phase 4.V\n")
    lines.append(f"Preselected doc_ids: {', '.join(d for _, d in SMOKE_DOCS)}\n")
    lines.append(f"k_step: {args.top_k}\n")

    for corpus, doc_id in SMOKE_DOCS:
        if corpus not in services:
            services[corpus] = RetrievalService.from_config(
                corpus=corpus,
                chroma_root=args.chroma_root,
                config_path=REPO / "config/experiment.yaml",
            )
        svc = services[corpus]
        result = svc.retrieve_for_batch(
            batch_doc_ids=[doc_id],
            k_step=args.top_k,
            domain=domains[corpus],
        )
        header = f"\n## {corpus} — {doc_id}  (k_step={result.k_step}, per_sub_query_k={result.per_sub_query_k})\n"
        lines.append(header)
        print(header.rstrip())
        for i, ch in enumerate(result.chunks, start=1):
            row = (
                f"{i:>2}. sub_q={ch.sub_query_index}  dist={ch.distance:.4f}  "
                f"chunk_id={ch.chunk_id}  offset={ch.metadata.get('offset_start')}-{ch.metadata.get('offset_end')}"
            )
            preview = f"    {_preview(ch.text)}"
            lines.append(row)
            lines.append(preview)
            print(row)
            print(preview)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
