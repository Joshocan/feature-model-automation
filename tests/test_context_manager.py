from __future__ import annotations

import os
from pathlib import Path
import pytest

from fame.utils.runtime import workspace
from fame.context import ContextManager, ContextBuildConfig, chunks_from_chunks_json


def test_context_manager_builds_and_appends() -> None:
    ws = workspace("preprocess", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    chunks_dir = paths.processed_data / "chunks"
    files = sorted(chunks_dir.glob("*.chunks.json"))
    if len(files) < 1:
        pytest.skip("No chunks.json found. Run ingestion first.")

    cm = ContextManager()
    cfg = ContextBuildConfig(max_total_chars=20_000, max_chunks=10)

    chunks1 = chunks_from_chunks_json(files[0])
    b1 = cm.add_initial_context(chunks1, cfg, title="INITIAL")
    assert "INITIAL" in b1
    assert len(cm.state.full_context()) > 100

    # adding same file again yields no new evidence in delta
    b2 = cm.add_delta_context(chunks1, cfg, title="DELTA")
    assert "no new evidence" in b2.lower()
