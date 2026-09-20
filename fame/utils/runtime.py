# fame/utils/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from fame.utils.dirs import FamePaths, build_paths, ensure_for_stage, ensure_dir


@dataclass(frozen=True)
class FameWorkspace:
    """
    Convenience wrapper to:
      1) resolve/build all standard FAME paths
      2) create only the directories required for a given stage
    """
    paths: FamePaths
    created: Dict[str, Path]
    stage: str


def workspace(stage: str, base_dir: Optional[str | Path] = None) -> FameWorkspace:
    """
    Build paths and create the directories needed for a specific stage.

    Example:
        ws = workspace("vectorize")
        chroma_dir = ws.paths.vector_db
    """
    p = build_paths(Path(base_dir) if base_dir is not None else None)
    created = ensure_for_stage(stage, p)
    return FameWorkspace(paths=p, created=created, stage=stage)


def ensure_base_only(base_dir: Optional[str | Path] = None) -> FameWorkspace:
    """
    Minimal bootstrap: create only the bare minimum that many runs assume exists.
    (Useful for 'init' commands or sanity checks.)

    Creates:
      - base_dir
      - data/raw
      - prompts
      - prompts/specifications
      - api_keys
      - results (root)
    """
    p = build_paths(Path(base_dir) if base_dir is not None else None)

    created: Dict[str, Path] = {}
    for label, d in {
        "BASE_DIR": p.base_dir,
        "RAW_DATA": p.raw_data,
        "PROMPTS": p.prompts,
        "SPECIFICATIONS": p.specifications,
        "API_KEYS": p.api_keys,
        "RESULTS": p.results,
    }.items():
        ensure_dir(d)
        created[label] = d

    return FameWorkspace(paths=p, created=created, stage="base")


def print_created(ws: FameWorkspace) -> None:
    """
    Pretty-print what was created for debugging.
    """
    print(f"\n=== FAME Workspace: stage='{ws.stage}' ===")
    print(f"BASE_DIR: {ws.paths.base_dir}")
    if not ws.created:
        print("(no directories created)")
    else:
        for k in sorted(ws.created.keys()):
            print(f"{k}: {ws.created[k]}")
    print("=================================\n")
