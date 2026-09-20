#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from fame.retrieval.service import RetrievalService


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run retrieval against Chroma collections for FAME RAG pipelines.")
    p.add_argument("--root-feature", required=True, help="Root feature name (e.g., 'Model Federation')")
    p.add_argument("--domain", required=True, help="Domain (e.g., 'Model-Driven Engineering')")
    p.add_argument("--collections", nargs="+", required=True, help="One or more Chroma collection names")
    p.add_argument("--k", type=int, default=int(os.getenv("RAG_K", "6")), help="Top-k per collection (default: 6)")
    p.add_argument("--max-total", type=int, default=int(os.getenv("RAG_MAX_TOTAL", "12")), help="Max total results (default: 12)")
    p.add_argument("--max-total-chars", type=int, default=int(os.getenv("RAG_MAX_CHARS", "18000")), help="Max total evidence chars")
    p.add_argument("--max-chunk-chars", type=int, default=int(os.getenv("RAG_MAX_CHUNK_CHARS", "2500")), help="Max chars per chunk")
    p.add_argument("--where", default="", help="Optional metadata filter as JSON string (advanced).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Optional metadata filter
    where = None
    if args.where.strip():
        import json
        where = json.loads(args.where)

    retr = RetrievalService(base_dir=os.getenv("FAME_BASE_DIR"))

    res = retr.retrieve(
        root_feature=args.root_feature,
        domain=args.domain,
        collections=args.collections,
        n_results_per_collection=args.k,
        max_total_results=args.max_total,
        where=where,
    )

    evidence = retr.to_prompt_evidence(
        res,
        max_total_chars=args.max_total_chars,
        max_chunk_chars=args.max_chunk_chars,
    )

    print("\n==================== RETRIEVAL ====================")
    print(f"CHROMA_MODE  : {os.getenv('CHROMA_MODE', 'persistent')}")
    print(f"CHROMA_PATH  : {os.getenv('CHROMA_PATH', '')}")
    print(f"CHROMA_HOST  : {os.getenv('CHROMA_HOST', '127.0.0.1')}")
    print(f"CHROMA_PORT  : {os.getenv('CHROMA_PORT', '8000')}")
    print("---------------------------------------------------")
    print("Query used:")
    print(res.query)
    print("---------------------------------------------------")
    print(f"Collections: {args.collections}")
    print(f"Results   : {len(res.chunks)}")
    print("===================================================\n")

    if not res.chunks:
        print("WARN:  No results. Check that collections exist and are indexed.")
        return

    print(evidence)
    print("\nSUCCESS: Done.")


if __name__ == "__main__":
    main()
