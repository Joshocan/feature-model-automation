from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import chromadb

from fame.utils.runtime import workspace
from fame.ingestion.pipeline import ingest_one_file
from fame.vectorization.pipeline import index_chunks_json
from fame.vectorization.chroma_indexer import ChromaConfig, connect_client


def _require_pdf(raw_dir: Path) -> Path:
    pdfs = list(raw_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No PDF in data/raw. Add at least one PDF to run this integration test.")
    return pdfs[0]


def _ollama_reachable() -> bool:
    import requests
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        return 200 <= r.status_code < 400
    except Exception:
        return False


def _chroma_reachable() -> bool:
    import requests
    host = os.getenv("CHROMA_HOST", "127.0.0.1")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    for url in [
        f"http://{host}:{port}/api/v1/heartbeat",
        f"http://{host}:{port}/heartbeat",
        f"http://{host}:{port}/",
    ]:
        try:
            r = requests.get(url, timeout=2)
            if 200 <= r.status_code < 400:
                return True
        except Exception:
            pass
    # persistent mode does not need server, so don't skip in that case
    return os.getenv("CHROMA_MODE", "persistent").strip().lower() == "persistent"


@pytest.mark.integration
def test_preprocessing_for_rag_end_to_end() -> None:
    """
    End-to-end integration test:
      PDF -> chunks.json -> embeddings -> indexed in Chroma
    """
    # Must have Ollama running for embedding
    if not _ollama_reachable():
        pytest.skip("Ollama is not reachable. Start it first (or run scripts/preprocessing_for_rag.py).")

    # Chroma: either http reachable OR persistent mode
    if not _chroma_reachable():
        pytest.skip("Chroma is not reachable. Start it first (or set CHROMA_MODE=persistent).")

    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    pdf = _require_pdf(paths.raw_data)

    # Ingest one file to a test folder (so it doesn't mix with your main run)
    test_chunks_dir = paths.processed_data / "test_chunks"
    test_chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks_json = ingest_one_file(pdf, out_dir=test_chunks_dir)
    assert chunks_json.exists()

    # Index into a unique test collection
    collection_name = f"test_ingestion_{int(time.time())}"
    res = index_chunks_json(chunks_json, collection=collection_name, batch_size=8)
    assert res["added"] > 0, f"Expected >0 docs added, got: {res}"

    # Verify collection count via Chroma client
    cfg = ChromaConfig.from_env(default_path=paths.vector_db)
    client = connect_client(cfg)
    col = client.get_collection(name=collection_name)
    count = col.count()
    assert count > 0, "Collection count is 0 after indexing"

    # Cleanup (optional): delete the test collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
