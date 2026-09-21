"""PDF → cleaned text extraction for the iFS 2027 chunking pipeline.

Uses ``unstructured.partition.pdf`` to obtain semantic elements
(Title / NarrativeText / ListItem / Footer / ...) rather than a flat character
stream. Element-type information lets us drop footers, page-number lines, and
figure captions before chunking.

Deterministic: same PDF and same unstructured version produce byte-identical
output. Not deterministic across unstructured versions — pin the version in
D12 (``data/encoder_versions.txt``) and record it in ``preprocessing_version``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List

# Reduce ONNX threading noise on shutdown
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Element type names we keep from an unstructured partition. Everything else
# (Footer, PageBreak, Image, Header) is discarded before chunking.
_KEEP_TYPES = frozenset({"NarrativeText", "Title", "ListItem", "UncategorizedText"})

# Match single-line page-number-like elements to drop even when categorised
# as NarrativeText.
_PAGENUM_RE = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdm]{1,6})\s*$", re.IGNORECASE)


def _element_texts(elements: Iterable) -> List[str]:
    """Extract text from unstructured elements, keeping only meaningful types."""
    parts: List[str] = []
    for elem in elements:
        type_name = type(elem).__name__
        if type_name not in _KEEP_TYPES:
            continue
        text = str(elem).strip()
        if not text or _PAGENUM_RE.match(text):
            continue
        # Collapse in-paragraph line breaks (unstructured preserves them
        # from the source PDF flow) so paragraph boundaries stay stable.
        text = re.sub(r"\s+", " ", text)
        parts.append(text)
    return parts


def extract_pdf_text(pdf_path: str | Path, *, strategy: str = "fast") -> str:
    """Extract cleaned, paragraph-normalised text from a single PDF.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    strategy : str
        Passed through to ``unstructured.partition.pdf.partition_pdf``.
        ``"fast"`` is deterministic and fast (~5s / 20-page PDF).
        ``"hi_res"`` engages layout models and OCR; slower and less determinstic.

    Returns
    -------
    str
        Cleaned text with double-newline paragraph separators. No front/back
        matter is stripped here — call :func:`fame.ingestion.cleaning.strip_matter`
        on the returned text before chunking.
    """
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "unstructured[pdf] is required. Install: pip install 'unstructured[pdf]'"
        ) from exc

    p = Path(pdf_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    elements = partition_pdf(filename=str(p), strategy=strategy)
    paragraphs = _element_texts(elements)
    return "\n\n".join(paragraphs)
