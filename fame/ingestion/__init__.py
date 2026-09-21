"""Ingestion package — PDF → cleaned text → deterministic chunks (D11)."""
from .chunking import Chunk, chunk_text
from .cleaning import (
    strip_front_matter,
    strip_back_matter,
    strip_matter,
    normalise_whitespace,
)
from .pdf_loader import extract_pdf_text
from .pipeline import (
    ChunkingConfig,
    DocReport,
    build_corpus_chunks,
    summarise_reports,
)
from .serialize import save_chunks_jsonl, load_chunks_jsonl

__all__ = [
    "Chunk",
    "chunk_text",
    "strip_front_matter",
    "strip_back_matter",
    "strip_matter",
    "normalise_whitespace",
    "extract_pdf_text",
    "ChunkingConfig",
    "DocReport",
    "build_corpus_chunks",
    "summarise_reports",
    "save_chunks_jsonl",
    "load_chunks_jsonl",
]
