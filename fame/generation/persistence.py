"""Atomic write helpers for run artefacts (Phase 5.8, 5.T8).

Every write is: write to sibling ``.tmp`` file → ``fsync`` → atomic ``rename``
so an interrupted run leaves either the previous complete state or no file at
all. Never a torn write. The loop calls these between steps so resume from
step j means every step 0..j-1 has been fully persisted.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunPaths:
    """Resolved paths for a single run's artefacts."""
    root:            Path         # results/<campaign>/<corpus>/<config_hash>/<run_id>/
    fm_gen:          Path         # D7  final feature model
    fm_iter_dir:     Path         # D8  per-step models
    context_log:     Path         # D9  jsonl
    run_meta:        Path         # D10 json

    @classmethod
    def for_run(cls, *, results_root: Path | str, campaign_id: str,
                corpus: str, config_hash: str, run_id: str) -> "RunPaths":
        base = (Path(results_root).expanduser().resolve()
                / campaign_id / corpus / config_hash / run_id)
        return cls(
            root=base,
            fm_gen=base / "fm_gen.xml",
            fm_iter_dir=base / "fm_iter",
            context_log=base / "context_log.jsonl",
            run_meta=base / "run_meta.json",
        )

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.fm_iter_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic writers
# ─────────────────────────────────────────────────────────────────────────────

def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically. Creates parents if missing."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding=encoding) as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def atomic_write_json(path: Path | str, obj: Any, *, indent: int = 2) -> None:
    text = json.dumps(obj, indent=indent, ensure_ascii=False, sort_keys=True)
    atomic_write_text(path, text)


def atomic_append_jsonl(path: Path | str, obj: Any) -> None:
    """Append one JSON-encoded record. Not atomic across records, but each
    record is written in one write() call which POSIX treats as atomic below
    PIPE_BUF (~4KB). Context-log records are well under that."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
