"""RetrievalService — batched, 4-sub-query, doc_id-filtered retrieval (Phase 4.7).

At step j with batch ``B_j = {doc_id_1, doc_id_2, ...}`` the service executes
each of the four fixed sub-queries against the corpus's Chroma collection with
a ``doc_id IN B_j`` filter, requests ``k_step / 4`` results per sub-query, then
merges and deduplicates by ``chunk_id``. Retrieval scores are preserved so a
downstream context-log (D9) can record them per chunk.

The service refuses to run if it detects unprefixed input at either end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from fame.vectorization.chroma_indexer import ChromaLocation, open_client
from fame.vectorization.embeddings import OllamaEmbedder, Embedder

from .query_templates import format_sub_query, load_sub_queries


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    sub_query_index: int    # which of the four sub-queries retrieved this chunk


@dataclass
class RetrievalResult:
    batch_doc_ids: List[str]
    k_step: int
    per_sub_query_k: List[int]
    chunks: List[EvidenceChunk] = field(default_factory=list)


class RetrievalService:
    """Per-corpus retriever. One instance per (corpus, chroma_root) pair."""

    def __init__(
        self,
        *,
        corpus: str,
        chroma_root: Path | str,
        sub_query_templates: Sequence[str],
        embedder: Embedder | None = None,
    ) -> None:
        if len(sub_query_templates) != 4:
            raise ValueError(
                f"RetrievalService requires exactly 4 sub-queries, got {len(sub_query_templates)}"
            )
        self.corpus = corpus
        self.sub_queries = list(sub_query_templates)
        self.embedder = embedder or OllamaEmbedder()
        self._location = ChromaLocation.for_corpus(chroma_root, corpus)
        self._client = open_client(self._location.path)
        self._collection = self._client.get_collection(name=self._location.collection_name)

    # ---------------------------------------------------------------- classmethod

    @classmethod
    def from_config(cls, *, corpus: str, chroma_root: Path | str,
                    config_path: Path | str) -> "RetrievalService":
        return cls(
            corpus=corpus,
            chroma_root=chroma_root,
            sub_query_templates=load_sub_queries(config_path),
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def split_k(k_step: int, n_sub_queries: int = 4,
                remainder_policy: str = "spread_first") -> List[int]:
        """Distribute k_step across n_sub_queries.

        ``spread_first`` (default): floor(k_step / n), extra 1s to the first
        ``k_step % n`` sub-queries.
        """
        if k_step <= 0:
            raise ValueError(f"k_step must be positive, got {k_step}")
        base, rem = divmod(k_step, n_sub_queries)
        if remainder_policy != "spread_first":
            raise ValueError(f"unknown remainder_policy: {remainder_policy}")
        return [base + (1 if i < rem else 0) for i in range(n_sub_queries)]

    # ---------------------------------------------------------------- public API

    def retrieve_for_batch(
        self,
        *,
        batch_doc_ids: Sequence[str],
        k_step: int,
        domain: str,
    ) -> RetrievalResult:
        """Run the four sub-queries with a doc_id-∈-batch filter.

        Deduplicates by ``chunk_id``, preserving the smallest distance and the
        earliest-firing sub-query index for each chunk.
        """
        if not batch_doc_ids:
            raise ValueError("batch_doc_ids must be non-empty")
        per_q = self.split_k(k_step, n_sub_queries=len(self.sub_queries))
        result = RetrievalResult(
            batch_doc_ids=list(batch_doc_ids),
            k_step=k_step,
            per_sub_query_k=per_q,
        )
        # Formatted (with search_query: prefix) queries
        formatted = [format_sub_query(q, domain=domain) for q in self.sub_queries]
        # Embed all four at once
        query_embeddings = self.embedder.embed_queries(formatted)

        where = {"doc_id": {"$in": list(batch_doc_ids)}}
        by_chunk: Dict[str, EvidenceChunk] = {}

        for i, (n_wanted, emb) in enumerate(zip(per_q, query_embeddings)):
            if n_wanted <= 0:
                continue
            hits = self._collection.query(
                query_embeddings=[emb],
                n_results=n_wanted,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            ids       = hits.get("ids",       [[]])[0]
            docs      = hits.get("documents", [[]])[0]
            metas     = hits.get("metadatas", [[]])[0]
            distances = hits.get("distances", [[]])[0]
            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                meta = meta or {}
                prev = by_chunk.get(cid)
                if prev is None or dist < prev.distance:
                    by_chunk[cid] = EvidenceChunk(
                        chunk_id=cid,
                        doc_id=meta.get("doc_id", ""),
                        text=doc,
                        metadata=meta,
                        distance=float(dist),
                        sub_query_index=(prev.sub_query_index if prev else i),
                    )

        # Rank ascending by distance
        result.chunks = sorted(by_chunk.values(), key=lambda c: c.distance)
        return result
