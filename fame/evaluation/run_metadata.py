from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class RunMetadata:
    pipeline_id: str
    llm_model: str
    iteration_id: int
    retrieval_enabled: bool
    top_k_chunks: int
    prompt_type: str
    model_temperature: float
    timestamp: str

    def to_dict(self):
        return self.__dict__


def write_run_metadata(meta: RunMetadata, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")


def default_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"

