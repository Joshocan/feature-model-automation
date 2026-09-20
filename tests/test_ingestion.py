from __future__ import annotations

import json
import os
from pathlib import Path

from fame.utils.runtime import workspace
from fame.ingestion.pipeline import ingest_one_file


def test_ingestion_loads_pdf_from_raw() -> None:
    """
    Integration test:
      - loads a real PDF from data/raw
      - runs ingestion (clean + chunk)
      - asserts chunks.json is created and non-empty
    """

    # Resolve workspace
    ws = workspace("preprocess", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    raw_dir = paths.raw_data
    out_dir = paths.processed_data / "chunks"

    assert raw_dir.exists(), "data/raw directory does not exist"

    pdfs = list(raw_dir.glob("*.pdf"))
    assert pdfs, "No PDF files found in data/raw to test ingestion"

    pdf = pdfs[0]  # take first available PDF

    # Run ingestion
    out_json = ingest_one_file(pdf, out_dir=out_dir)

    # Assertions
    assert out_json.exists(), "chunks.json file was not created"

    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert "chunks" in payload, "Output JSON missing 'chunks'"
    assert isinstance(payload["chunks"], list), "'chunks' is not a list"
    assert len(payload["chunks"]) > 0, "No chunks produced from PDF"

    assert payload.get("num_chunks") == len(payload["chunks"]), \
        "num_chunks does not match actual number of chunks"

    # Optional sanity check on text
    first_text = payload["chunks"][0].get("text", "")
    assert isinstance(first_text, str)
    assert len(first_text.strip()) > 20, "Chunk text too short (unexpected)"

    print(f"SUCCESS: Ingestion test passed for: {pdf.name}")
