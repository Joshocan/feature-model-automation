from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from fame.utils.runtime import workspace
from fame.utils.dirs import ensure_dir

from .discovery import list_input_files
from .loaders import load_and_clean
from .chunking import partition_and_chunk, simple_chunk
from .serialize import save_chunks_json


PathLike = Union[str, Path]


def ingest_one_file(
    file_path: PathLike,
    out_dir: Optional[PathLike] = None,
) -> Path:
    """
    Ingest + prepare ONE file:
      - load + clean
      - chunk
      - save chunks json

    Returns the path to the created chunks.json.
    """
    ws = workspace("preprocess")  # ensures processed_data exists
    paths = ws.paths

    fp = Path(file_path).expanduser().resolve()
    if not fp.exists():
        raise FileNotFoundError(f"Input file not found: {fp}")

    chunks_out_dir = Path(out_dir).expanduser().resolve() if out_dir else (paths.processed_data / "chunks")
    ensure_dir(chunks_out_dir)

    _PLAIN_TEXT_EXTS = {".txt", ".md", ".markdown", ".yaml", ".yml", ".json",
                        ".sql", ".ddl", ".graphql", ".gql", ".xml", ".xsd"}
    cleaned = load_and_clean(fp)
    if fp.suffix.lower() in _PLAIN_TEXT_EXTS:
        chunks = simple_chunk(cleaned, source_filename=fp.name)
    else:
        chunks = partition_and_chunk(cleaned, source_filename=fp.name)
    out_json = save_chunks_json(chunks, source_filename=fp.name, output_dir=chunks_out_dir)
    return out_json


def ingest_and_prepare(
    raw_dir: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
) -> Dict[str, List[Path]]:
    """
    End-to-end ingestion + preparation (batch):
      1) discover files in data/raw
      2) ingest each file using ingest_one_file()
      3) save chunks to JSON under processed_data/chunks (default)

    Returns:
      {
        "processed": [list of chunks.json paths],
        "skipped": [list of skipped input files]
      }
    """
    ws = workspace("preprocess")  # creates processed_data + chunk_reports (from dirs.py)
    paths = ws.paths

    raw_path = Path(raw_dir).expanduser().resolve() if raw_dir else paths.raw_data
    chunks_out_dir = Path(out_dir).expanduser().resolve() if out_dir else (paths.processed_data / "chunks")

    ensure_dir(raw_path)        # ensure input folder exists
    ensure_dir(chunks_out_dir)  # ensure output folder exists

    inputs = list_input_files(raw_path)
    processed: List[Path] = []
    skipped: List[Path] = []

    if not inputs:
        return {"processed": processed, "skipped": skipped}

    for fp in inputs:
        try:
            out_json = ingest_one_file(fp, out_dir=chunks_out_dir)
            processed.append(out_json)
        except Exception as e:
            print(f"WARN:  Skipping {Path(fp).name}: {e}")
            skipped.append(Path(fp))

    return {"processed": processed, "skipped": skipped}


if __name__ == "__main__":
    result = ingest_and_prepare()
    print(f"SUCCESS: Ingestion complete. Processed: {len(result['processed'])}, Skipped: {len(result['skipped'])}")
    if result["processed"]:
        print("Example output:", result["processed"][0])
