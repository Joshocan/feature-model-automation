from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_chunks_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Chunks JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def extract_chunks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Invalid chunks payload: 'chunks' must be a list")
    return chunks


def normalize_chunk_record(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure each chunk has the keys we need for indexing:
      - chunk_id (id)
      - text (document)
      - metadata (dict)
    """
    chunk_id = str(chunk.get("chunk_id", "")).strip()
    text = (chunk.get("text") or "").strip()
    metadata = chunk.get("metadata") or {}

    if not chunk_id:
        raise ValueError("Chunk missing 'chunk_id'")
    if not text:
        raise ValueError(f"Chunk '{chunk_id}' has empty text")
    if not isinstance(metadata, dict):
        metadata = {}

    # keep original 'source' as metadata too
    source = str(chunk.get("source", "")).strip()
    if source:
        metadata.setdefault("source", source)

    # Chroma expects scalar metadata values; coerce complex types to strings.
    for k, v in list(metadata.items()):
        if isinstance(v, (str, int, float, bool)) or v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            metadata[k] = ", ".join(str(item) for item in v)
        else:
            metadata[k] = str(v)

    return {"chunk_id": chunk_id, "text": text, "metadata": metadata}
