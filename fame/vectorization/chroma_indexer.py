"""Chroma persistence + collection helpers for the iFS 2027 campaign.

Two collections total — one per corpus — at ``data/chroma/{corpus}/``. Each
collection stores one row per chunk, keyed by the deterministic ``chunk_id``
from D11. Metadata carries ``doc_id``, character offsets, and the preprocessing
version so a corrupted/stale index is detectable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import chromadb

from .embeddings import Embedder


@dataclass(frozen=True)
class ChromaLocation:
    path: Path
    collection_name: str

    @staticmethod
    def for_corpus(chroma_root: Path | str, corpus: str) -> "ChromaLocation":
        """Deterministic path/name per corpus. One directory, one collection."""
        return ChromaLocation(
            path=Path(chroma_root).expanduser().resolve() / corpus,
            collection_name=corpus,
        )


def open_client(path: Path | str) -> chromadb.PersistentClient:
    """Open (or create) a persistent Chroma client at ``path``."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(p))


def reset_collection(client: chromadb.PersistentClient, name: str,
                     metadata: Dict[str, Any] | None = None):
    """Ensure a fresh collection: delete if present, then create."""
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    return client.create_collection(name=name, metadata=metadata or {})


def _batches(seq: Sequence[Any], size: int) -> List[Sequence[Any]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def upsert_chunks(
    collection,
    *,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[Dict[str, Any]],
    embedder: Embedder,
    batch_size: int = 32,
) -> Tuple[int, int]:
    """Embed and upsert chunks in fixed-size batches.

    Documents are passed to ``embedder.embed_documents`` which enforces the
    ``search_document:`` prefix. The **canonical text** stored in the ``documents``
    field is the raw chunk text (unchanged) — the prefix is applied only at
    embed time, and the embedder is idempotent if the prefix is already present.

    Returns ``(num_added, num_failed)``.
    """
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("ids, documents, metadatas must have equal length")

    added, failed = 0, 0
    for b_ids, b_docs, b_meta in zip(
        _batches(ids, batch_size),
        _batches(documents, batch_size),
        _batches(metadatas, batch_size),
    ):
        try:
            embeddings = embedder.embed_documents(b_docs)
            collection.upsert(
                ids=list(b_ids),
                documents=list(b_docs),
                metadatas=list(b_meta),
                embeddings=embeddings,
            )
            added += len(b_ids)
        except Exception as exc:
            print(f"WARN: batch upsert of {len(b_ids)} items failed: {exc}")
            failed += len(b_ids)
    return added, failed
