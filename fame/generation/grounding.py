"""Grounding — assemble the CONTEXT block per step.

* **Non-RAG:** every chunk of ``B_j``, in deterministic chunk_id order, no
  selection, no k.
* **RAG:** batch-scoped retrieval via the 4 fixed sub-queries with a
  ``doc_id ∈ B_j`` filter; merged, deduplicated by chunk_id, ranked by
  distance.

Both arms format chunks identically so the prompt text differs only in the
subset of chunks included. This isolates H1b (selection effect) from any
formatting confound.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

from fame.retrieval import RetrievalService


@dataclass(frozen=True)
class ChunkEvidence:
    """Uniform chunk representation for both grounding arms."""
    chunk_id: str
    doc_id: str
    text: str
    offset_start: int
    offset_end: int
    # RAG-only fields:
    distance: Optional[float] = None
    sub_query_index: Optional[int] = None


@dataclass
class GroundingContext:
    """The context block for one step."""
    grounding: str                 # "rag" | "nonrag"
    batch_doc_ids: List[str]
    chunks: List[ChunkEvidence]    # ordered as they appear in the prompt
    k_step: Optional[int]          # None for nonrag
    per_sub_query_k: Optional[List[int]]     # None for nonrag


def _load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_nonrag(
    *,
    chunks_jsonl: Path,
    batch_doc_ids: Sequence[str],
) -> GroundingContext:
    all_chunks = _load_chunks_jsonl(chunks_jsonl)
    in_batch = [c for c in all_chunks if c["doc_id"] in set(batch_doc_ids)]
    # Deterministic order: chunk_id lexicographic (already stable across runs).
    in_batch.sort(key=lambda c: c["chunk_id"])
    evidence = [
        ChunkEvidence(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            text=c["text"],
            offset_start=c["offsets"][0],
            offset_end=c["offsets"][1],
        )
        for c in in_batch
    ]
    return GroundingContext(
        grounding="nonrag",
        batch_doc_ids=list(batch_doc_ids),
        chunks=evidence,
        k_step=None,
        per_sub_query_k=None,
    )


def _build_rag(
    *,
    retrieval_service: RetrievalService,
    batch_doc_ids: Sequence[str],
    k_doc: int,
    domain: str,
) -> GroundingContext:
    k_step = k_doc * len(batch_doc_ids)
    result = retrieval_service.retrieve_for_batch(
        batch_doc_ids=list(batch_doc_ids),
        k_step=k_step,
        domain=domain,
    )
    evidence = [
        ChunkEvidence(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            text=c.text,
            offset_start=int(c.metadata.get("offset_start", 0)),
            offset_end=int(c.metadata.get("offset_end", 0)),
            distance=c.distance,
            sub_query_index=c.sub_query_index,
        )
        for c in result.chunks
    ]
    return GroundingContext(
        grounding="rag",
        batch_doc_ids=list(batch_doc_ids),
        chunks=evidence,
        k_step=k_step,
        per_sub_query_k=result.per_sub_query_k,
    )


def build_grounding(
    *,
    grounding: str,
    batch_doc_ids: Sequence[str],
    domain: str,
    chunks_jsonl: Path | str,
    retrieval_service: Optional[RetrievalService] = None,
    k_doc: Optional[int] = None,
) -> GroundingContext:
    """Dispatcher — build the ``GroundingContext`` for one step."""
    chunks_path = Path(chunks_jsonl).expanduser().resolve()
    if grounding == "nonrag":
        return _build_nonrag(chunks_jsonl=chunks_path, batch_doc_ids=batch_doc_ids)
    if grounding == "rag":
        if retrieval_service is None or k_doc is None:
            raise ValueError("RAG requires retrieval_service and k_doc")
        return _build_rag(
            retrieval_service=retrieval_service,
            batch_doc_ids=batch_doc_ids,
            k_doc=k_doc,
            domain=domain,
        )
    raise ValueError(f"unknown grounding: {grounding!r} (expected 'rag' or 'nonrag')")


def build_grounding_from_record(
    *,
    record: Dict[str, Any],
    chunks_jsonl: Path | str,
) -> GroundingContext:
    """Rehydrate the exact evidence selection recorded for a failed call.

    Resume must not repeat vector retrieval: approximate-nearest-neighbour
    ordering can vary even when the index and query are unchanged. Text is
    loaded from the frozen canonical chunk store and ordered by recorded IDs.
    """
    all_chunks = _load_chunks_jsonl(Path(chunks_jsonl).expanduser().resolve())
    by_id = {c["chunk_id"]: c for c in all_chunks}
    chunk_ids = list(record.get("chunk_ids") or [])
    doc_ids = list(record.get("chunk_doc_ids") or [])
    scores = list(record.get("retrieval_scores") or [])
    sub_queries = list(record.get("sub_query_indices") or [])
    if not chunk_ids or not (len(chunk_ids) == len(doc_ids) == len(scores) == len(sub_queries)):
        raise ValueError("recorded RAG context has inconsistent chunk metadata")
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in by_id]
    if missing:
        raise ValueError(f"recorded RAG chunks are absent from the frozen store: {missing}")
    evidence: List[ChunkEvidence] = []
    for chunk_id, doc_id, distance, sub_query_index in zip(
            chunk_ids, doc_ids, scores, sub_queries):
        chunk = by_id[chunk_id]
        if chunk["doc_id"] != doc_id:
            raise ValueError(f"recorded doc_id mismatch for chunk {chunk_id}")
        evidence.append(ChunkEvidence(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=chunk["text"],
            offset_start=chunk["offsets"][0],
            offset_end=chunk["offsets"][1],
            distance=distance,
            sub_query_index=sub_query_index,
        ))
    return GroundingContext(
        grounding=str(record.get("grounding")),
        batch_doc_ids=list(record.get("batch_doc_ids") or []),
        chunks=evidence,
        k_step=record.get("k_step"),
        per_sub_query_k=record.get("per_sub_query_k"),
    )


def format_context_text(gc: GroundingContext) -> str:
    """Serialise the chunks as they will appear in the prompt CONTEXT block.

    Identical formatting across arms — only chunk selection differs.
    """
    parts: List[str] = []
    for i, c in enumerate(gc.chunks, start=1):
        header = f"[EVIDENCE {i}] doc_id={c.doc_id} chunk_id={c.chunk_id}"
        parts.append(header + "\n" + c.text)
    return "\n\n".join(parts)
