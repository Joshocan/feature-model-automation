"""Write one JSON candidate per web-run repetition."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


def emit_web_candidate(result: Mapping[str, str], ordinal: int) -> Path:
    """Copy the pipeline's JSON SystemModel output into FAME web's stable contract."""
    source = result.get("fm_xml") or result.get("final_xml")
    if not source:
        raise RuntimeError("Pipeline result did not include a model output path.")
    raw = Path(source).read_text(encoding="utf-8").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].rstrip()
    try:
        model = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Pipeline output is not valid SystemModel JSON: {source}") from exc
    if not isinstance(model, dict):
        raise RuntimeError("Pipeline output must be a JSON object.")

    output_root = Path(os.environ.get("FAME_OUTPUT_DIR", "results")).expanduser().resolve()
    candidates_dir = output_root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    target = candidates_dir / f"candidate-{ordinal:03d}.json"
    target.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    producer = {
        "llmProvider": os.environ.get("FAME_LLM_PROVIDER", "ollama"),
        "llmModel": os.environ.get("FAME_LLM_MODEL", "unknown") or "unknown",
    }
    meta_path = result.get("meta")
    if meta_path:
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                model_name = meta.get("llm_model")
                if isinstance(model_name, str) and model_name.strip():
                    producer["llmModel"] = model_name
        except (OSError, json.JSONDecodeError):
            pass
    target.with_suffix(".producer.json").write_text(json.dumps(producer, indent=2) + "\n", encoding="utf-8")
    # A stable machine-readable event consumed by FAME Web. Keep it on stdout
    # so normal CLI users can still see candidate completion progress.
    print(f"FAME_WEB_PROGRESS candidate_completed={ordinal}", flush=True)
    return target
