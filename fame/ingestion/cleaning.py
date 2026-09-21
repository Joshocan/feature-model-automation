"""Text-cleaning helpers for the iFS 2027 chunking pipeline.

The functions here operate on plain strings and are deliberately conservative:
they only cut where a keyword anchor is confidently matched, and never modify
canonical content. The chunker downstream tracks character offsets into the
returned text, so any cleaning step must be idempotent and length-stable
across identical input.
"""
from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# Section-level stripping (front/back matter)
# ─────────────────────────────────────────────────────────────────────────────

_FRONT_ANCHOR_RE = re.compile(
    r"^\s*(abstract|introduction|1\s*[\.\s]+\s*introduction)\b",
    re.IGNORECASE | re.MULTILINE,
)

_BACK_ANCHOR_RE = re.compile(
    r"^\s*(references|bibliography|acknowledge?ments?|"
    r"appendix|author.{0,3}bio|about\s+the\s+authors?)\b",
    re.IGNORECASE | re.MULTILINE,
)


def strip_front_matter(text: str) -> str:
    """Cut everything before the first ``Abstract`` or ``Introduction`` heading.

    Front matter typically holds copyright blocks, journal boilerplate, and
    author affiliations. These are dense with domain vocabulary that retrieves
    well but contributes nothing to the target feature model.

    If no anchor is found, the input is returned unchanged.
    """
    if not text:
        return ""
    m = _FRONT_ANCHOR_RE.search(text)
    if not m:
        return text
    return text[m.start(1):]


def strip_back_matter(text: str) -> str:
    """Cut everything at and after the first ``References``, ``Bibliography``,
    ``Acknowledgments``, ``Appendix``, or author-bio heading.

    Reference lists inflate provenance L2 attribution with citations that are
    true but meaningless (see IFS brief §1). Stripping them before chunking
    keeps them out of retrieval entirely.
    """
    if not text:
        return ""
    m = _BACK_ANCHOR_RE.search(text)
    if not m:
        return text
    return text[:m.start()].rstrip()


def strip_matter(text: str) -> str:
    """Convenience: apply :func:`strip_front_matter` then :func:`strip_back_matter`."""
    return strip_back_matter(strip_front_matter(text))


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers retained for compatibility with older tests
# ─────────────────────────────────────────────────────────────────────────────

def remove_reference_section(text: str) -> str:
    """Legacy alias for :func:`strip_back_matter`."""
    return strip_back_matter(text)


def remove_inline_citations(text: str) -> str:
    """Remove numeric ``[3]`` / ``(3)`` and author-year ``(Smith 2020)`` citations.

    Retained for callers that want to strip inline citations. The primary
    chunking path in this project **does not** call this: retaining inline
    citations keeps sentence boundaries intact and preserves how the source
    reads. Use with care.
    """
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]|\(\d+(?:,\s*\d+)*\)", "", text)
    text = re.sub(r"\([^)]*?\d{4}[^)]*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalise_whitespace(text: str) -> str:
    """Collapse consecutive whitespace but preserve paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)
    return text.strip()


def clean_noise(text: str) -> str:
    """Legacy full-clean pipeline used by the older four-pipeline code.

    The new N-loop pipeline calls :func:`strip_matter` directly and does not
    normalise citations. This function is retained so older callers keep working.
    """
    if not isinstance(text, str):
        return ""
    text = remove_reference_section(text)
    text = remove_inline_citations(text)
    text = normalise_whitespace(text)
    return text
