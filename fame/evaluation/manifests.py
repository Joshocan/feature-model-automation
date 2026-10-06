"""Validated document identities shared by offline evaluation drivers."""
import csv
from pathlib import Path


def read_manifest_doc_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        if not reader.fieldnames or "doc_id" not in reader.fieldnames:
            raise ValueError(f"{path}: required doc_id column missing (expected semicolon-separated CSV)")
        for row in reader:
            doc = (row.get("doc_id") or "").strip()
            if not doc:
                raise ValueError(f"{path}:{reader.line_num}: blank doc_id")
            if doc in seen:
                raise ValueError(f"{path}:{reader.line_num}: duplicate doc_id {doc!r}")
            seen.add(doc)
            ids.append(doc)
    if not ids:
        raise ValueError(f"{path}: no document IDs found")
    return ids
