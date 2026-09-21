"""End-to-end ingestion pipeline for the iFS 2027 campaign.

Per-corpus orchestration:
    PDF  ── unstructured ──►  raw paragraphs
                          ──►  strip_matter (front + back)
                          ──►  chunk_text (sliding window)
                          ──►  chunks.jsonl (D11)

Deterministic. Same inputs + pinned unstructured version produce the same
``chunk_id`` set and byte-identical text spans.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .cleaning import strip_matter
from .chunking import chunk_text, Chunk
from .pdf_loader import extract_pdf_text
from .serialize import save_chunks_jsonl


@dataclass(frozen=True)
class ChunkingConfig:
    max_chunk_chars: int = 1500
    overlap_chars:   int = 225        # 15% of 1500
    preprocessing_version: str = "v2-1500-15"
    pdf_strategy:    str = "fast"


@dataclass
class DocReport:
    doc_id:        str
    pdf_path:      str
    n_elements:    int
    text_chars:    int
    n_chunks:      int
    seconds:       float
    error:         Optional[str] = None


def _read_doc_ids_from_manifest(manifest_csv: str | Path) -> List[Tuple[str, str]]:
    """Return ``(doc_id, current_filename)`` pairs from a semicolon manifest."""
    rows = list(csv.DictReader(open(manifest_csv), delimiter=";"))
    return [(r["doc_id"], r["current_filename"]) for r in rows]


def build_corpus_chunks(
    *,
    corpus: str,
    manifest_csv: str | Path,
    raw_dir: str | Path,
    out_jsonl: str | Path,
    config: ChunkingConfig = ChunkingConfig(),
    on_doc_start: Optional[Callable[[str], None]] = None,
    on_doc_done:  Optional[Callable[[DocReport], None]] = None,
) -> List[DocReport]:
    """Build ``chunks.jsonl`` for one corpus.

    Iterates the manifest in order, extracts each PDF, strips front/back
    matter, chunks with the frozen policy, and appends to the JSONL.
    Returns a per-document report suitable for chunks_stats.json.

    Truncates ``out_jsonl`` before writing.
    """
    out_path = Path(out_jsonl).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w").close()   # truncate; will append per doc below

    raw = Path(raw_dir).expanduser().resolve()
    reports: List[DocReport] = []

    for doc_id, filename in _read_doc_ids_from_manifest(manifest_csv):
        if on_doc_start:
            on_doc_start(doc_id)
        pdf = raw / filename
        t0 = time.time()
        try:
            raw_text = extract_pdf_text(pdf, strategy=config.pdf_strategy)
            cleaned = strip_matter(raw_text)
            chunks: List[Chunk] = chunk_text(
                cleaned,
                doc_id=doc_id,
                max_chunk_chars=config.max_chunk_chars,
                overlap_chars=config.overlap_chars,
                preprocessing_version=config.preprocessing_version,
            )
            save_chunks_jsonl(chunks, out_path, append=True)
            report = DocReport(
                doc_id=doc_id,
                pdf_path=str(pdf),
                n_elements=raw_text.count("\n\n") + 1,      # rough paragraph count
                text_chars=len(cleaned),
                n_chunks=len(chunks),
                seconds=round(time.time() - t0, 2),
            )
        except Exception as exc:
            report = DocReport(
                doc_id=doc_id,
                pdf_path=str(pdf),
                n_elements=0,
                text_chars=0,
                n_chunks=0,
                seconds=round(time.time() - t0, 2),
                error=f"{type(exc).__name__}: {exc}",
            )
        reports.append(report)
        if on_doc_done:
            on_doc_done(report)

    return reports


def summarise_reports(corpus: str, reports: List[DocReport]) -> Dict[str, object]:
    """Compute chunks-per-doc summary statistics for a corpus."""
    import statistics as st
    successes = [r for r in reports if r.error is None and r.n_chunks > 0]
    counts = [r.n_chunks for r in successes]
    return {
        "corpus": corpus,
        "docs_total":  len(reports),
        "docs_ok":     len(successes),
        "docs_failed": len(reports) - len(successes),
        "min":         min(counts) if counts else 0,
        "median":      int(st.median(counts)) if counts else 0,
        "mean":        round(st.mean(counts), 1) if counts else 0.0,
        "max":         max(counts) if counts else 0,
        "total_chunks": sum(counts),
        "seconds_total": round(sum(r.seconds for r in reports), 1),
        "errors": [{"doc_id": r.doc_id, "error": r.error}
                   for r in reports if r.error is not None],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compat shims for the older four-pipeline entry points.
# These are kept only until Phase 5 removes their callers.
# ─────────────────────────────────────────────────────────────────────────────

def ingest_one_file(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "The legacy ingest_one_file(...) path was removed in Phase 4. "
        "Use build_corpus_chunks(...) instead."
    )


def ingest_and_prepare(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "The legacy ingest_and_prepare(...) path was removed in Phase 4. "
        "Use build_corpus_chunks(...) instead."
    )
