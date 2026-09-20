from __future__ import annotations

import os
from pathlib import Path

import pytest

import chromadb

from fame.utils.runtime import workspace
from fame.retrieval.service import RetrievalService
from fame.vectorization.chroma_indexer import ChromaConfig, connect_client


def _list_collections(client) -> list[str]:
    # Chroma python client differences across versions:
    # - some have list_collections()
    # - some return objects with .name
    cols = []
    try:
        got = client.list_collections()
        for c in got:
            name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
            if name:
                cols.append(str(name))
    except Exception:
        pass
    return cols


@pytest.mark.integration
def test_retrieval_returns_results() -> None:
    """
    Integration test:
      - assumes vectorization already indexed at least one collection
      - runs retrieval with the default query template
      - asserts we get at least 1 evidence chunk
    """
    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    cfg = ChromaConfig.from_env(default_path=paths.vector_db)
    client = connect_client(cfg)

    cols = _list_collections(client)
    if not cols:
        pytest.skip(
            "No Chroma collections found. Run vectorization first "
            "(scripts/run_vectorization.py or scripts/preprocessing_for_rag.py)."
        )

    # Use first 1-3 collections for test (works for SS/MS/IS)
    use_cols = cols[:3]

    retr = RetrievalService(base_dir=os.getenv("FAME_BASE_DIR"))
    res = retr.retrieve(
        root_feature=os.getenv("TEST_ROOT_FEATURE", "Model Federation"),
        domain=os.getenv("TEST_DOMAIN", "Model-Driven Engineering"),
        collections=use_cols,
        n_results_per_collection=int(os.getenv("TEST_RAG_K", "3")),
        max_total_results=int(os.getenv("TEST_RAG_MAX_TOTAL", "6")),
    )

    assert res.query and isinstance(res.query, str)
    assert len(res.chunks) > 0, (
        "Retrieval returned 0 chunks. "
        "Check that collections contain documents and embeddings."
    )

    # sanity: each evidence chunk should have required fields
    for ch in res.chunks:
        assert ch.chunk_id
        assert ch.text and len(ch.text.strip()) > 10
