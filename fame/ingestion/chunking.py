from __future__ import annotations

import re
from typing import Any, Dict, List

_TARGET_CHUNK_CHARS = 1_200
_MAX_CHUNK_CHARS = 2_500


def _split_long_paragraph(text: str) -> List[str]:
    """Split an oversized paragraph on sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    parts: List[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > _MAX_CHUNK_CHARS:
            parts.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf:
        parts.append(buf.strip())
    return parts or [text]


def simple_chunk(cleaned_text: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Paragraph-based chunker with no external dependencies.
    Used as fallback when unstructured is not installed, and as the
    primary chunker for plain-text artefact types (.md, .yaml, .json,
    .sql, .graphql, .xml, etc.).
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', cleaned_text) if p.strip()]

    segments: List[str] = []
    for para in paragraphs:
        if len(para) > _MAX_CHUNK_CHARS:
            segments.extend(_split_long_paragraph(para))
        else:
            segments.append(para)

    chunks: List[str] = []
    buf = ""
    for seg in segments:
        if buf and len(buf) + len(seg) + 2 > _TARGET_CHUNK_CHARS:
            chunks.append(buf.strip())
            buf = seg
        else:
            buf = (buf + "\n\n" + seg).strip() if buf else seg
    if buf:
        chunks.append(buf.strip())

    out: List[Dict[str, Any]] = []
    for i, text in enumerate(chunks):
        out.append({
            "chunk_id": f"{source_filename}::chunk::{i}",
            "source": source_filename,
            "text": text,
            "metadata": {"filename": source_filename},
        })
    return out


def partition_and_chunk(cleaned_text: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Use unstructured.partition_text + chunk_by_title to create chunks.
    Returns JSON-serializable dict chunks.
    """
    try:
        from unstructured.partition.text import partition_text
        from unstructured.chunking.title import chunk_by_title
    except Exception:
        return simple_chunk(cleaned_text, source_filename)

    elements = partition_text(text=cleaned_text)
    chunks = chunk_by_title(elements)

    out: List[Dict[str, Any]] = []
    for i, el in enumerate(chunks):
        text = getattr(el, "text", "") or ""
        meta = getattr(el, "metadata", None)

        # metadata can be a dataclass-like object; keep it JSON-safe
        meta_dict: Dict[str, Any] = {}
        if meta is not None:
            # best-effort: many unstructured metadata objects implement to_dict()
            if hasattr(meta, "to_dict"):
                try:
                    meta_dict = meta.to_dict()  # type: ignore[attr-defined]
                except Exception:
                    meta_dict = {}
            else:
                # fallback: pick common attrs if present
                for k in ("filename", "page_number", "category", "coordinates", "languages"):
                    if hasattr(meta, k):
                        meta_dict[k] = getattr(meta, k)

        out.append(
            {
                "chunk_id": f"{source_filename}::chunk::{i}",
                "source": source_filename,
                "text": text,
                "metadata": meta_dict,
            }
        )

    return out
