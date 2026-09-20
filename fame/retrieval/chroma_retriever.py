from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb


@dataclass(frozen=True)
class ChromaConn:
    mode: str                # "persistent" or "http"
    path: Path               # used in persistent mode
    host: str
    port: int

    @staticmethod
    def from_env(default_path: str | Path) -> "ChromaConn":
        mode = os.getenv("CHROMA_MODE", "persistent").strip().lower()
        host = os.getenv("CHROMA_HOST", "127.0.0.1").strip()
        port = int(os.getenv("CHROMA_PORT", "8000").strip())
        path = Path(os.getenv("CHROMA_PATH", str(default_path))).expanduser().resolve()
        return ChromaConn(mode=mode, path=path, host=host, port=port)


def connect(conn: ChromaConn):
    if conn.mode == "http":
        return chromadb.HttpClient(host=conn.host, port=conn.port)
    conn.path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(conn.path))


@dataclass
class RawChromaQueryResult:
    collection: str
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


def _flatten_one(res: Dict[str, Any], collection: str) -> RawChromaQueryResult:
    """
    Chroma returns nested lists: ids=[[...]...]
    Here we assume we query one query_text at a time, so take index 0.
    """
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    # normalize
    ids = [str(x) for x in (ids or [])]
    docs = [str(x) for x in (docs or [])]
    metas = [x if isinstance(x, dict) else {} for x in (metas or [])]
    dists = [float(x) for x in (dists or [])]

    # align lengths safely
    n = min(len(ids), len(docs), len(metas), len(dists))
    return RawChromaQueryResult(
        collection=collection,
        ids=ids[:n],
        documents=docs[:n],
        metadatas=metas[:n],
        distances=dists[:n],
    )


def query_collection(
    client,
    collection_name: str,
    query_text: str,
    n_results: int = 8,
    where: Optional[Dict[str, Any]] = None,
    query_embeddings: Optional[List[float]] = None,
) -> RawChromaQueryResult:
    col = client.get_collection(name=collection_name)
    if query_embeddings is not None:
        res = col.query(
            query_embeddings=[query_embeddings],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    else:
        res = col.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    return _flatten_one(res, collection=collection_name)


def query_many_collections(
    client,
    collections: Sequence[str],
    query_text: str,
    n_results_per_collection: int = 6,
    where: Optional[Dict[str, Any]] = None,
    query_embeddings: Optional[List[float]] = None,
) -> List[RawChromaQueryResult]:
    out: List[RawChromaQueryResult] = []
    for c in collections:
        try:
            out.append(
                query_collection(
                    client,
                    collection_name=c,
                    query_text=query_text,
                    n_results=n_results_per_collection,
                    where=where,
                    query_embeddings=query_embeddings,
                )
            )
        except Exception as e:
            print(f"WARN:  Retrieval skipped collection '{c}': {e}")
    return out
