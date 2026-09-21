"""Vectorization — embed D11 chunks and populate Chroma per corpus."""
from .chroma_indexer import (
    ChromaLocation,
    open_client,
    reset_collection,
    upsert_chunks,
)
from .embeddings import DOC_PREFIX, QUERY_PREFIX, Embedder, OllamaEmbedder
from .pipeline import IndexBuildReport, build_index

__all__ = [
    "ChromaLocation",
    "DOC_PREFIX",
    "Embedder",
    "IndexBuildReport",
    "OllamaEmbedder",
    "QUERY_PREFIX",
    "build_index",
    "open_client",
    "reset_collection",
    "upsert_chunks",
]
