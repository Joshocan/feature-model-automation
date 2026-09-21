"""Corpus indexing orchestrator (Phase 4.6).

Reads a canonical ``chunks.jsonl`` (D11), embeds each chunk with the mandatory
``search_document:`` prefix, and writes to a persistent Chroma collection.

One collection per corpus. The collection is **reset** on every call so a
re-index is deterministic and can never accumulate stale entries from a
previous preprocessing version.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .chroma_indexer import (
    ChromaLocation,
    open_client,
    reset_collection,
    upsert_chunks,
)
from .embeddings import OllamaEmbedder, Embedder


@dataclass
class IndexBuildReport:
    corpus: str
    location: ChromaLocation
    chunks_read: int
    chunks_upserted: int
    chunks_failed: int


def _load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_index(
    *,
    corpus: str,
    chunks_jsonl: str | Path,
    chroma_root: str | Path,
    embedder: Optional[Embedder] = None,
    batch_size: int = 32,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> IndexBuildReport:
    """Build the Chroma collection for one corpus from its chunks.jsonl."""
    chunks_jsonl = Path(chunks_jsonl).expanduser().resolve()
    if not chunks_jsonl.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_jsonl}")

    chunks = _load_chunks_jsonl(chunks_jsonl)
    location = ChromaLocation.for_corpus(chroma_root, corpus)
    client = open_client(location.path)
    coll = reset_collection(
        client,
        name=location.collection_name,
        metadata={
            "corpus":                 corpus,
            "chunks_source":          str(chunks_jsonl),
            "preprocessing_version":  (chunks[0]["preprocessing_version"] if chunks else ""),
            "embedding_model":        "nomic-embed-text",
            "document_prefix":        "search_document: ",
        },
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "doc_id":                c["doc_id"],
            "offset_start":          c["offsets"][0],
            "offset_end":            c["offsets"][1],
            "preprocessing_version": c["preprocessing_version"],
        }
        for c in chunks
    ]

    embedder = embedder or OllamaEmbedder()

    added_total = 0
    failed_total = 0
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        added, failed = upsert_chunks(
            coll,
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embedder=embedder,
            batch_size=batch_size,
        )
        added_total += added
        failed_total += failed
        if on_progress:
            on_progress(end, len(ids))

    return IndexBuildReport(
        corpus=corpus,
        location=location,
        chunks_read=len(chunks),
        chunks_upserted=added_total,
        chunks_failed=failed_total,
    )
