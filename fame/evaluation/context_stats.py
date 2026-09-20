from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

from fame.utils.context_budget import compute_max_total_chars, estimate_max_chars_from_tokens
from fame.context import EvidenceChunk
import re


@dataclass(frozen=True)
class ContextStats:
    model: str
    total_chars: int
    estimated_tokens: int
    max_tokens: int
    max_chars: int
    utilization_tokens: float
    utilization_chars: float
    num_chunks: int
    num_sources: int
    sources: Sequence[str]


def estimate_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    if not text:
        return 0
    return int(len(text) / chars_per_token)


def analyze_context_usage(
    *,
    model: str,
    context_text: str,
    chunks: Optional[Sequence[EvidenceChunk]] = None,
    max_tokens: Optional[int] = None,
    model_windows: Optional[Dict[str, int]] = None,
    chars_per_token: float = 4.0,
    safety: float = 0.8,
) -> ContextStats:
    """
    Compute rough context usage for a given model.
    If max_tokens is not provided, uses model_windows (or defaults).
    """
    total_chars = len(context_text or "")
    est_tokens = estimate_tokens(context_text, chars_per_token=chars_per_token)

    if max_tokens is None:
        max_chars = compute_max_total_chars(
            model,
            model_windows=model_windows,
            chars_per_token=chars_per_token,
            safety=safety,
        )
        max_tokens = int(max_chars / chars_per_token) if max_chars else 0
    else:
        max_chars = estimate_max_chars_from_tokens(
            max_tokens, chars_per_token=chars_per_token, safety=safety
        )

    utilization_tokens = (est_tokens / max_tokens) if max_tokens else 0.0
    utilization_chars = (total_chars / max_chars) if max_chars else 0.0

    sources_set: Set[str] = set()
    num_chunks = 0
    if chunks:
        num_chunks = len(chunks)
        for c in chunks:
            if c.source:
                sources_set.add(str(c.source))
            else:
                src = c.metadata.get("source") if isinstance(c.metadata, dict) else ""
                if src:
                    sources_set.add(str(src))

    return ContextStats(
        model=model,
        total_chars=total_chars,
        estimated_tokens=est_tokens,
        max_tokens=max_tokens,
        max_chars=max_chars,
        utilization_tokens=utilization_tokens,
        utilization_chars=utilization_chars,
        num_chunks=num_chunks,
        num_sources=len(sources_set),
        sources=sorted(sources_set),
    )


def extract_prompt_sources(prompt_text: str) -> Tuple[int, Sequence[str]]:
    """
    Parse SS-NonRAG prompt text to count chunks and unique sources.
    Expects chunk headers like: [CHUNK 1] id=... source=Article_1 ...
    """
    if not prompt_text:
        return 0, []
    chunk_count = len(re.findall(r"^\[CHUNK\s+\d+\]", prompt_text, flags=re.MULTILINE))
    sources = set(re.findall(r"\bsource=([^\s]+)", prompt_text))
    return chunk_count, sorted(sources)


def analyze_prompt_usage(
    *,
    model: str,
    prompt_text: str,
    max_tokens: Optional[int] = None,
    model_windows: Optional[Dict[str, int]] = None,
    chars_per_token: float = 4.0,
    safety: float = 0.8,
) -> ContextStats:
    """
    Compute stats from the fully-rendered prompt text.
    """
    chunk_count, sources = extract_prompt_sources(prompt_text)
    base = analyze_context_usage(
        model=model,
        context_text=prompt_text,
        max_tokens=max_tokens,
        model_windows=model_windows,
        chars_per_token=chars_per_token,
        safety=safety,
    )
    return ContextStats(
        model=base.model,
        total_chars=base.total_chars,
        estimated_tokens=base.estimated_tokens,
        max_tokens=base.max_tokens,
        max_chars=base.max_chars,
        utilization_tokens=base.utilization_tokens,
        utilization_chars=base.utilization_chars,
        num_chunks=chunk_count,
        num_sources=len(sources),
        sources=sources,
    )
