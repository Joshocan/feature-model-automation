from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    source: str = ""
    score: Optional[float] = None  # distance or relevance (optional)


def load_chunks_json(chunks_json: PathLike) -> Dict[str, Any]:
    p = Path(chunks_json).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Chunks JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def chunks_from_chunks_json(chunks_json: PathLike) -> List[EvidenceChunk]:
    payload = load_chunks_json(chunks_json)
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Invalid chunks.json: 'chunks' must be a list")

    out: List[EvidenceChunk] = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("chunk_id", "")).strip()
        text = (c.get("text") or "").strip()
        meta = c.get("metadata") or {}
        src = c.get("source") or meta.get("source") or payload.get("source") or ""

        if not cid or not text:
            continue
        if not isinstance(meta, dict):
            meta = {}

        out.append(EvidenceChunk(chunk_id=cid, text=text, metadata=meta, source=str(src)))
    return out
