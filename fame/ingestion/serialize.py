"""Chunk serialisation — canonical JSONL for the iFS 2027 campaign."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def save_chunks_jsonl(
    chunks: Iterable[Dict[str, Any] | Any],
    out_path: str | Path,
    *,
    append: bool = False,
) -> Path:
    """Write chunks as newline-delimited JSON (D11 canonical format).

    Each line is one JSON object matching the ``Chunk.to_dict()`` schema:
    ``{chunk_id, doc_id, offsets, text, preprocessing_version}``.

    Parameters
    ----------
    chunks : Iterable
        Chunks to write. Accepts :class:`fame.ingestion.chunking.Chunk`
        instances (has ``to_dict()``) or plain dicts.
    out_path : str | Path
        Target ``chunks.jsonl`` path. Parent directories are created.
    append : bool
        If ``False`` (default) the file is truncated first. Use ``True`` when
        streaming chunks across multiple corpus files.
    """
    p = Path(out_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(p, mode, encoding="utf-8") as fh:
        for c in chunks:
            payload = c.to_dict() if hasattr(c, "to_dict") else c
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")
    return p


def load_chunks_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a chunks.jsonl file into a list of dicts."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {p}")
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# Legacy helper — kept until any remaining callers are migrated.
def save_chunks_json(chunks: List[Dict[str, Any]], source_filename: str, output_dir: str | Path) -> Path:
    """Legacy per-file JSON writer used by the old four-pipeline code."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(source_filename).name}.chunks.json"
    payload = {"source": source_filename, "num_chunks": len(chunks), "chunks": chunks}
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file
