from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple


SUPPORTED_EXTS: Tuple[str, ...] = (
    ".pdf", ".txt", ".docx",
    ".md", ".markdown",
    ".yaml", ".yml",
    ".json",
    ".sql", ".ddl",
    ".graphql", ".gql",
    ".xml", ".xsd",
)


def list_input_files(raw_dir: str | Path, exts: Iterable[str] = SUPPORTED_EXTS) -> List[Path]:
    """
    List supported files in raw_dir (non-recursive by default).
    """
    raw = Path(raw_dir).expanduser().resolve()
    if not raw.exists():
        return []

    extset = {e.lower() for e in exts}
    files = [p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in extset]
    return sorted(files)
