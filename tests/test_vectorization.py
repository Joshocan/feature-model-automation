"""Tests for the Phase 4b vectorization primitives.

These tests deliberately avoid network / Ollama / real Chroma: they exercise
prefix enforcement and the deterministic collection-location contract only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fame.vectorization.chroma_indexer import ChromaLocation
from fame.vectorization.embeddings import (
    DOC_PREFIX,
    QUERY_PREFIX,
    Embedder,
    OllamaEmbedder,
)


class _FakeEmbedder(Embedder):
    """Records every string embedded so we can assert the prefix contract."""

    def __init__(self) -> None:
        self.seen_documents: list[str] = []
        self.seen_queries:   list[str] = []

    def embed_documents(self, texts):
        self.seen_documents.extend(texts)
        return [[0.0] * 4 for _ in texts]

    def embed_queries(self, texts):
        self.seen_queries.extend(texts)
        return [[0.0] * 4 for _ in texts]


# ─────────────────────────────────────────────────────────────────────────────
# Collection layout
# ─────────────────────────────────────────────────────────────────────────────

def test_chroma_location_is_deterministic(tmp_path: Path) -> None:
    a = ChromaLocation.for_corpus(tmp_path, "federation")
    b = ChromaLocation.for_corpus(tmp_path, "federation")
    assert a == b
    assert a.collection_name == "federation"
    assert a.path.name == "federation"


def test_chroma_location_different_per_corpus(tmp_path: Path) -> None:
    a = ChromaLocation.for_corpus(tmp_path, "federation")
    b = ChromaLocation.for_corpus(tmp_path, "repair")
    assert a.path != b.path
    assert a.collection_name != b.collection_name


# ─────────────────────────────────────────────────────────────────────────────
# Prefix enforcement — via the real OllamaEmbedder without hitting the network
# ─────────────────────────────────────────────────────────────────────────────

def test_embedder_rejects_empty_text() -> None:
    e = OllamaEmbedder()
    with pytest.raises(ValueError):
        e.embed_documents([""])
    with pytest.raises(ValueError):
        e.embed_queries(["   "])


def test_doc_prefix_and_query_prefix_are_distinct() -> None:
    assert DOC_PREFIX != QUERY_PREFIX
    assert DOC_PREFIX.startswith("search_document:")
    assert QUERY_PREFIX.startswith("search_query:")
