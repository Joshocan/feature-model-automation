from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def save_chunks_json(chunks: List[Dict[str, Any]], source_filename: str, output_dir: str | Path) -> Path:
    """
    Save chunks to <output_dir>/<source_filename>.chunks.json
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(source_filename).name
    out_file = out_dir / f"{safe_name}.chunks.json"

    payload = {
        "source": safe_name,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file
