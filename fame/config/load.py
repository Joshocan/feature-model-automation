# fame/config/load.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fame.utils.dirs import resolve_base_dir
from .schema import FameConfig, load_yaml_config, parse_config


def load_config(config_path: Optional[str | Path] = None) -> FameConfig:
    """
    Load config from:
      1) provided config_path
      2) env FAME_CONFIG (if set)
      3) default: config/fame.yaml (repo root)
    """
    repo_root = resolve_base_dir()

    env_path = os.getenv("FAME_CONFIG", "").strip()
    if config_path is None and env_path:
        config_path = env_path

    if config_path is None:
        config_path = repo_root / "config" / "fame.yaml"

    doc = load_yaml_config(config_path)
    return parse_config(doc, repo_root=repo_root)
