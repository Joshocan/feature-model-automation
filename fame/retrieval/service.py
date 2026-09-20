from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fame.utils.runtime import workspace

from .query_templates import DEFAULT_RAG_QUERY_TEMPLATE, QueryContext, build_query
from .chroma_retriever import (
    ChromaConn,
    RawChromaQueryResult,
    connect,
    query_collection,
    query_many_collections,
)
from fame.vectorization.embeddings import OllamaEmbedder


@dataclass(frozen=True)
class EvidenceChunk:
    collection: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: List[EvidenceChunk]


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars].rstrip() + "…"


def format_evidence_for_prompt(
    chunks: Sequence[EvidenceChunk],
    max_total_chars: int = 18_000,
    max_chunk_chars: int = 2_500,
) -> str:
    """
    Create a prompt-friendly evidence block with provenance.
    """
    parts: List[str] = []
    total = 0

    for i, ch in enumerate(chunks, start=1):
        header = f"[EVIDENCE {i}] collection={ch.collection} chunk_id={ch.chunk_id} distance={ch.distance:.4f}"
        meta_src = ch.metadata.get("source") or ch.metadata.get("filename") or ""
        if meta_src:
            header += f" source={meta_src}"
        header += "\n"

        body = _truncate(ch.text, max_chunk_chars)
        block = header + body + "\n"

        if total + len(block) > max_total_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts).strip()


class RetrievalService:
    """
    Pipeline-agnostic retriever for SS/MS/IS-RGFM.

    - Builds query from your default template (or a custom one)
    - Queries one or many Chroma collections
    - Returns ranked EvidenceChunk list + helper to format for prompts
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        ws = workspace("vectorize", base_dir=base_dir or os.getenv("FAME_BASE_DIR"))
        self.paths = ws.paths

        conn = ChromaConn.from_env(default_path=self.paths.vector_db)
        self.client = connect(conn)
        self.conn = conn

    def build_default_query(self, root_feature: str, domain: str) -> str:
        ctx = QueryContext(root_feature=root_feature, domain=domain)
        return build_query(ctx, template=DEFAULT_RAG_QUERY_TEMPLATE)

    def retrieve(
        self,
        *,
        root_feature: str,
        domain: str,
        collections: Sequence[str],
        query_template: str = DEFAULT_RAG_QUERY_TEMPLATE,
        n_results_per_collection: int = 6,
        where: Optional[Dict[str, Any]] = None,
        max_total_results: int = 12,
    ) -> RetrievalResult:
        """
        Retrieve evidence for SS/MS/IS-RGFM.

        collections:
          - SS-RGFM: [collection_for_single_article]
          - MS-RGFM: [collectionA, collectionB, collectionC]
          - IS-RGFM: you can pass just the current article, or a growing list

        where:
          - optional Chroma metadata filter
        """
        ctx = QueryContext(root_feature=root_feature, domain=domain)
        query = build_query(ctx, template=query_template)
        embedder = OllamaEmbedder()
        query_embedding = embedder.embed_documents([query])[0]

        # Query each collection and merge
        raw_results = query_many_collections(
            self.client,
            collections=collections,
            query_text=query,
            n_results_per_collection=n_results_per_collection,
            where=where,
            query_embeddings=query_embedding,
        )

        all_chunks: List[EvidenceChunk] = []
        for r in raw_results:
            for cid, doc, meta, dist in zip(r.ids, r.documents, r.metadatas, r.distances):
                all_chunks.append(
                    EvidenceChunk(
                        collection=r.collection,
                        chunk_id=cid,
                        text=doc,
                        metadata=meta or {},
                        distance=dist,
                    )
                )

        # Rank by distance (lower is more similar in Chroma’s default)
        all_chunks.sort(key=lambda x: x.distance)

        # De-duplicate by chunk_id across collections (best-effort)
        seen = set()
        deduped: List[EvidenceChunk] = []
        for ch in all_chunks:
            if ch.chunk_id in seen:
                continue
            seen.add(ch.chunk_id)
            deduped.append(ch)
            if len(deduped) >= max_total_results:
                break

        return RetrievalResult(query=query, chunks=deduped)

    def to_prompt_evidence(
        self,
        result: RetrievalResult,
        max_total_chars: int = 18_000,
        max_chunk_chars: int = 2_500,
    ) -> str:
        return format_evidence_for_prompt(
            result.chunks,
            max_total_chars=max_total_chars,
            max_chunk_chars=max_chunk_chars,
        )
