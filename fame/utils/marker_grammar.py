"""Provenance marker grammar for the iFS 2027 campaign.

Marker format frozen in Phase 2.2. Every generated <description> ends with
exactly one marker of the form:

    [src: id1, id2]

Rules:
  - Enclosed in "[src: " and "]".
  - One or more doc_ids, comma+space separated.
  - Identifiers exactly as given in the CONTEXT_DOC_IDS block.
  - Nothing after the closing bracket.

This parser is deterministic (no fuzzy matching) so downstream provenance
levels L0/L1/L2 can be computed unambiguously.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# One canonical regex. Anchored at end-of-string; deliberately strict.
_MARKER_RE = re.compile(r"\[src:\s+([A-Za-z0-9_,\s]+?)\]\s*\Z")


@dataclass(frozen=True)
class MarkerParseResult:
    """Outcome of parsing a single description."""

    doc_ids: List[str]
    marker_found: bool
    marker_text: Optional[str]  # raw text of the marker, e.g. "[src: rep_01, rep_17]"


def parse_marker(description_text: str) -> MarkerParseResult:
    """Parse the trailing [src: ...] marker from a description.

    Returns marker_found=False when no marker is present. The parser is
    strict: any deviation from the frozen format (missing "src:", wrong
    separator, trailing text) fails parseably.
    """
    if not description_text:
        return MarkerParseResult(doc_ids=[], marker_found=False, marker_text=None)
    m = _MARKER_RE.search(description_text)
    if not m:
        return MarkerParseResult(doc_ids=[], marker_found=False, marker_text=None)
    raw = m.group(1)
    ids = [tok.strip() for tok in raw.split(",")]
    ids = [tok for tok in ids if tok]
    return MarkerParseResult(
        doc_ids=ids,
        marker_found=True,
        marker_text=m.group(0),
    )
