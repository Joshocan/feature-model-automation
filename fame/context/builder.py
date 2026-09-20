from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .sources import EvidenceChunk


@dataclass(frozen=True)
class ContextBuildConfig:
    max_total_chars: int = 120_000   # big default for Non-RAG
    max_chunk_chars: int = 6_000
    max_chunks: int = 80
    include_headers: bool = True
    include_metadata: bool = False  # keep False by default (token bloat)
    order: str = "as_is"            # as_is | by_page_then_id | by_id


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def _sort(chunks: List[EvidenceChunk], order: str) -> List[EvidenceChunk]:
    if order == "by_id":
        return sorted(chunks, key=lambda c: c.chunk_id)

    if order == "by_page_then_id":
        def key(c: EvidenceChunk):
            page = c.metadata.get("page_number", c.metadata.get("page", 0))
            try:
                page = int(page)
            except Exception:
                page = 0
            return (page, c.chunk_id)
        return sorted(chunks, key=key)

    return chunks


def build_context(
    chunks: Sequence[EvidenceChunk],
    cfg: ContextBuildConfig = ContextBuildConfig(),
    title: str = "EVIDENCE",
) -> str:
    """
    Builds a single prompt context block (string).
    """
    items = _sort(list(chunks), cfg.order)

    parts: List[str] = []
    total = 0
    used = 0

    # top header (optional)
    header0 = f"=== {title} ===\n"
    parts.append(header0)
    total += len(header0)

    for i, ch in enumerate(items, start=1):
        if used >= cfg.max_chunks:
            break

        body = _truncate(ch.text, cfg.max_chunk_chars)

        header = ""
        if cfg.include_headers:
            header = f"[CHUNK {i}] id={ch.chunk_id}"
            if ch.source:
                header += f" source={ch.source}"
            if ch.score is not None:
                header += f" score={ch.score:.4f}"
            page = ch.metadata.get("page_number", ch.metadata.get("page", ""))
            if page != "":
                header += f" page={page}"
            header += "\n"

        meta_block = ""
        if cfg.include_metadata and ch.metadata:
            meta_block = f"metadata={ch.metadata}\n"

        block = header + meta_block + body + "\n\n"
        if total + len(block) > cfg.max_total_chars:
            break

        parts.append(block)
        total += len(block)
        used += 1

    return "".join(parts).strip()
