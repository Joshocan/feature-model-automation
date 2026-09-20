from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fame.utils.runtime import workspace
from fame.utils.dirs import ensure_dir

from .chunks_loader import load_chunks_json, extract_chunks, normalize_chunk_record
from .embeddings import OllamaEmbedder
from .chroma_indexer import ChromaConfig, connect_client, get_or_create_collection, upsert_chunks


PathLike = Union[str, Path]


def default_collection_name(chunks_json_path: Path) -> str:
    """
    Stable-ish default: <filename-without-suffix> (and strip extra '.pdf' if present).
    Example: paper.pdf.chunks.json -> paper
    """
    name = chunks_json_path.name
    # remove .chunks.json
    if name.endswith(".chunks.json"):
        name = name[: -len(".chunks.json")]
    # remove final .pdf if present
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.replace(" ", "_")


def index_chunks_json(
    chunks_json: PathLike,
    collection: Optional[str] = None,
    batch_size: int = 24,
) -> Dict[str, Any]:
    """
    Read a single <file>.chunks.json and index its chunks into Chroma.

    Env vars:
      - OLLAMA_HOST / OLLAMA_EMBED_MODEL
      - CHROMA_MODE (persistent|http), CHROMA_PATH, CHROMA_HOST, CHROMA_PORT
    """
    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    chunks_json_path = Path(chunks_json).expanduser().resolve()
    if not chunks_json_path.exists():
        raise FileNotFoundError(f"Chunks JSON not found: {chunks_json_path}")

    # default chroma path is your vector_db_dir
    cfg = ChromaConfig.from_env(default_path=paths.vector_db)
    client = connect_client(cfg)

    payload = load_chunks_json(chunks_json_path)
    raw_chunks = extract_chunks(payload)

    normalized: List[Dict[str, Any]] = []
    for c in raw_chunks:
        try:
            normalized.append(normalize_chunk_record(c))
        except Exception as e:
            print(f"WARN:  Skipping invalid chunk: {e}")

    if not normalized:
        return {"collection": collection or default_collection_name(chunks_json_path), "added": 0, "failed": 0}

    ids = [c["chunk_id"] for c in normalized]
    docs = [c["text"] for c in normalized]
    metas = []
    for c in normalized:
        m = dict(c["metadata"])
        m.setdefault("embedding_model", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
        metas.append(m)

    col_name = collection or default_collection_name(chunks_json_path)
    col = get_or_create_collection(client, col_name, metadata={"source_chunks": str(chunks_json_path.name)})

    embedder = OllamaEmbedder()  # uses env vars if set
    added, failed = upsert_chunks(col, ids=ids, documents=docs, metadatas=metas, embedder=embedder, batch_size=batch_size)

    return {"collection": col_name, "added": added, "failed": failed, "chroma_mode": cfg.mode}


def index_all_chunks(
    chunks_dir: Optional[PathLike] = None,
    batch_size: int = 24,
    collection_prefix: str = "",
) -> Dict[str, Any]:
    """
    Index all *.chunks.json under processed_data/chunks.

    collection name = <collection_prefix><default_collection_name(file)>
    """
    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    default_dir = paths.processed_data / "chunks"
    d = Path(chunks_dir).expanduser().resolve() if chunks_dir else default_dir
    ensure_dir(d)

    files = sorted(d.glob("*.chunks.json"))
    results: List[Dict[str, Any]] = []

    for f in files:
        col = f"{collection_prefix}{default_collection_name(f)}"
        r = index_chunks_json(f, collection=col, batch_size=batch_size)
        results.append(r)

    return {"indexed_files": len(files), "results": results}


def index_all_chunks_one_collection(
    chunks_dir: Optional[PathLike] = None,
    batch_size: int = 24,
    collection_name: str = "fame_all",
) -> Dict[str, Any]:
    """
    Index all *.chunks.json into a single collection (one_collection mode).
    """
    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    default_dir = paths.processed_data / "chunks"
    d = Path(chunks_dir).expanduser().resolve() if chunks_dir else default_dir
    ensure_dir(d)

    files = sorted(d.glob("*.chunks.json"))
    if not files:
        return {"collection": collection_name, "indexed_files": 0, "added": 0, "failed": 0}

    cfg = ChromaConfig.from_env(default_path=paths.vector_db)
    client = connect_client(cfg)
    col = get_or_create_collection(client, collection_name, metadata={"source_chunks": "one_collection"})

    embedder = OllamaEmbedder()

    total_added = 0
    total_failed = 0
    per_file: List[Dict[str, Any]] = []

    for f in files:
        payload = load_chunks_json(f)
        raw_chunks = extract_chunks(payload)
        normalized: List[Dict[str, Any]] = []
        for c in raw_chunks:
            try:
                n = normalize_chunk_record(c)
                # prefix chunk_id to avoid collisions across files
                n["chunk_id"] = f"{f.stem}:{n['chunk_id']}"
                normalized.append(n)
            except Exception as e:
                print(f"WARN:  Skipping invalid chunk from {f.name}: {e}")
        if not normalized:
            per_file.append({"file": f.name, "added": 0, "failed": 0})
            continue

        ids = [c["chunk_id"] for c in normalized]
        docs = [c["text"] for c in normalized]
        metas = []
        for c in normalized:
            m = dict(c["metadata"])
            m.setdefault("embedding_model", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
            m["source_file"] = f.name
            metas.append(m)

        added, failed = upsert_chunks(col, ids=ids, documents=docs, metadatas=metas, embedder=embedder, batch_size=batch_size)
        total_added += added
        total_failed += failed
        per_file.append({"file": f.name, "added": added, "failed": failed})

    return {
        "collection": collection_name,
        "indexed_files": len(files),
        "added": total_added,
        "failed": total_failed,
        "files": per_file,
        "chroma_mode": cfg.mode,
    }
