from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb

from .embeddings import Embedder


@dataclass
class ChromaConfig:
    mode: str  # "persistent" or "http"
    path: Path
    host: str
    port: int

    @staticmethod
    def from_env(default_path: str | Path) -> "ChromaConfig":
        """
        Env vars:
          - CHROMA_MODE: persistent|http  (default: persistent)
          - CHROMA_PATH: persistent storage path
          - CHROMA_HOST / CHROMA_PORT: for http mode
        """
        mode = os.getenv("CHROMA_MODE", "persistent").strip().lower()
        host = os.getenv("CHROMA_HOST", "127.0.0.1").strip()
        port = int(os.getenv("CHROMA_PORT", "8000").strip())
        path = Path(os.getenv("CHROMA_PATH", str(default_path))).expanduser().resolve()
        return ChromaConfig(mode=mode, path=path, host=host, port=port)


def connect_client(cfg: ChromaConfig):
    if cfg.mode == "http":
        return chromadb.HttpClient(host=cfg.host, port=cfg.port)
    # default persistent
    cfg.path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(cfg.path))


def get_or_create_collection(client, name: str, metadata: Optional[Dict[str, Any]] = None):
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(name=name, metadata=metadata or {})


def chunk_batches(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def upsert_chunks(
    collection,
    *,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[Dict[str, Any]],
    embedder: Embedder,
    batch_size: int = 32,
) -> Tuple[int, int]:
    """
    Upsert documents to Chroma with embeddings computed via embedder.
    Returns (num_added, num_failed)
    """
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("ids, documents, metadatas must have same length")

    added = 0
    failed = 0

    id_batches = chunk_batches(ids, batch_size)
    doc_batches = chunk_batches(documents, batch_size)
    meta_batches = chunk_batches(metadatas, batch_size)

    for b_ids, b_docs, b_meta in zip(id_batches, doc_batches, meta_batches):
        try:
            embeddings = embedder.embed_documents(b_docs)
            # If any embedding is empty, fail fast (better debugging)
            if any((not e) for e in embeddings):
                raise RuntimeError("One or more embeddings were empty. Check Ollama server/model.")

            collection.upsert(
                ids=list(b_ids),
                documents=list(b_docs),
                metadatas=list(b_meta),
                embeddings=embeddings,
            )
            added += len(b_ids)
        except Exception as e:
            print(f"WARN:  Failed batch upsert ({len(b_ids)} items): {e}")
            failed += len(b_ids)

    return added, failed
