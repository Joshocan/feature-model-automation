"""Sliding-window chunker for the iFS 2027 campaign (Phase 4.3, 4.4).

Semantics
---------
* Fixed target size: ``max_chunk_chars`` (frozen to 1500 in Phase 2.1).
* Fixed overlap:     ``overlap_chars``   (15% of max = 225 chars).
* Boundary preference (within the last 50% of the window):
    1. paragraph break  ``\\n\\n``
    2. sentence terminator  ``.`` / ``!`` / ``?`` followed by whitespace
* Only when neither is available in the second half of the window do we
  hard-cut at ``max_chunk_chars``.

Determinism
-----------
For a given ``(text, doc_id, max_chunk_chars, overlap_chars,
preprocessing_version)``, this function produces byte-identical output. The
``chunk_id`` is the first 16 hex characters of
SHA-256("doc_id|start|end|preprocessing_version"), so the same span in the
same source gets the same id across regenerations.

Offsets
-------
``offsets = [start, end]`` are half-open character indices into the **input
text** (i.e., the string after front/back-matter stripping). Recovery is
therefore ``text[start:end]``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


_SENTENCE_TERMINATORS = (". ", ".\n", "! ", "!\n", "? ", "?\n")


@dataclass(frozen=True)
class Chunk:
    """A single chunk emitted by :func:`chunk_text`."""
    chunk_id: str
    doc_id: str
    offsets: List[int]              # [start, end] into the input text
    text: str
    preprocessing_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _chunk_id(doc_id: str, start: int, end: int, preprocessing_version: str) -> str:
    """Deterministic 16-hex-char chunk id."""
    key = f"{doc_id}|{start}|{end}|{preprocessing_version}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


def _best_break(text: str, lo: int, hi: int) -> int:
    """Return the best boundary index in ``[lo, hi)``, or ``hi`` for a hard cut.

    Prefers a paragraph break; falls back to a sentence terminator; then a
    hard cut. The boundary must fall in the second half of the window to
    avoid pathologically small chunks when the source has short paragraphs.
    """
    if hi <= lo:
        return hi
    para = text.rfind("\n\n", lo, hi)
    if para > lo:
        return para
    best = -1
    for term in _SENTENCE_TERMINATORS:
        idx = text.rfind(term, lo, hi)
        if idx > best:
            best = idx + 1                # keep the terminator with the chunk
    return best if best > lo else hi


def chunk_text(
    text: str,
    *,
    doc_id: str,
    max_chunk_chars: int = 1500,
    overlap_chars: int = 225,
    preprocessing_version: str = "v2-1500-15",
) -> List[Chunk]:
    """Chunk ``text`` deterministically.

    Returns an empty list for empty or all-whitespace input.
    """
    if not text or not text.strip():
        return []
    if max_chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chunk_chars:
        raise ValueError(
            f"invalid chunker parameters: "
            f"max_chunk_chars={max_chunk_chars}, overlap_chars={overlap_chars}"
        )

    n = len(text)
    out: List[Chunk] = []
    start = 0

    # The boundary preference should only kick in in the second half of
    # the window, so pathological short paragraphs at chunk start don't
    # collapse the chunk. Use max/2 as the "acceptable break" floor.
    half = max_chunk_chars // 2

    while start < n:
        end = min(start + max_chunk_chars, n)
        if end < n:
            end = _best_break(text, start + half, end)

        span = text[start:end].strip()
        if span:
            cid = _chunk_id(doc_id, start, end, preprocessing_version)
            out.append(Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                offsets=[start, end],
                text=span,
                preprocessing_version=preprocessing_version,
            ))

        if end >= n:
            break
        # Slide forward. Overlap = each next chunk begins ``overlap_chars``
        # before the previous chunk's end.
        next_start = end - overlap_chars
        if next_start <= start:
            # Safety: never go backwards or stall
            next_start = start + max(1, max_chunk_chars - overlap_chars)
        start = next_start

    return out
