from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Set

from .builder import ContextBuildConfig, build_context
from .sources import EvidenceChunk


@dataclass
class ContextState:
    seen_chunk_ids: Set[str] = field(default_factory=set)
    blocks: List[str] = field(default_factory=list)

    def full_context(self) -> str:
        return "\n\n".join([b for b in self.blocks if b.strip()]).strip()


class ContextManager:
    """
    Reusable across:
      - Non-RAG: big in-learning context + iterative deltas
      - IS-RGFM: per-article deltas appended each iteration
      - MS-RGFM: initial context from multiple sources then deltas
      - SS-RGFM: just one context build
    """

    def __init__(self) -> None:
        self.state = ContextState()

    def add_initial_context(self, chunks: Sequence[EvidenceChunk], cfg: ContextBuildConfig, title: str) -> str:
        # mark all as seen
        for c in chunks:
            self.state.seen_chunk_ids.add(c.chunk_id)

        block = build_context(chunks, cfg=cfg, title=title)
        self.state.blocks.append(block)
        return block

    def add_delta_context(self, chunks: Sequence[EvidenceChunk], cfg: ContextBuildConfig, title: str) -> str:
        fresh: List[EvidenceChunk] = []
        for c in chunks:
            if c.chunk_id in self.state.seen_chunk_ids:
                continue
            self.state.seen_chunk_ids.add(c.chunk_id)
            fresh.append(c)

        if not fresh:
            block = f"=== {title} ===\n(no new evidence)\n"
            self.state.blocks.append(block)
            return block

        block = build_context(fresh, cfg=cfg, title=title)
        self.state.blocks.append(block)
        return block
