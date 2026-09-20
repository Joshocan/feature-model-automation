#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
from fame.vectorization.pipeline import index_all_chunks, index_all_chunks_one_collection
from fame.utils.runtime import workspace


def main() -> None:
    ap = argparse.ArgumentParser(description="Vectorize chunks into Chroma collections")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("VEC_BATCH_SIZE", "24")))
    ap.add_argument("--collection-mode", choices=["per_source", "one_collection"], default=os.getenv("VEC_COLLECTION_MODE", "per_source"))
    ap.add_argument("--one-collection-name", default=os.getenv("VEC_ONE_COLLECTION_NAME", "fame_all"))
    ap.add_argument("--collection-prefix", default=os.getenv("VEC_COLLECTION_PREFIX", ""))
    args = ap.parse_args()

    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    chunks_dir = args.chunks_dir or ""

    print("=== FAME Vectorization Stage ===")
    print(f"Base dir     : {paths.base_dir}")
    print(f"Chunks dir   : {chunks_dir or (paths.processed_data / 'chunks')}")
    print(f"Vector DB dir: {paths.vector_db}")
    print(f"Collection mode : {args.collection_mode}")
    print("===============================")

    if args.collection_mode == "one_collection":
        res = index_all_chunks_one_collection(
            chunks_dir=chunks_dir or None,
            batch_size=args.batch_size,
            collection_name=args.one_collection_name,
        )
    else:
        res = index_all_chunks(
            chunks_dir=chunks_dir or None,
            batch_size=args.batch_size,
            collection_prefix=args.collection_prefix,
        )

    print("\nSUCCESS: Vectorization complete.")
    print(res)


if __name__ == "__main__":
    main()
