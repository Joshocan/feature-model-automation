"""Pre-flight token counting + infeasibility guard (Phase 5.6).

Every step, we compute the assembled prompt's token count using a per-model
tokenizer. If the count exceeds the model's context window, we do **not** call
the model — the step is recorded as ``feasible=false`` and the loop advances.

Native tokenizers:
* ``tiktoken`` for OpenAI models (gpt-6-astra → o200k_base or model default)
* ``ollama`` HTTP ``/api/tokenize`` for local/cloud Ollama models
* Char/4 universal fallback for anything unrecognised
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

import requests


CHAR_PER_TOKEN_APPROX = 4.0     # universal English-prose fallback


def universal_estimate(text: str) -> int:
    """Character/4 heuristic. Safe upper bound for context-window guards."""
    if not text:
        return 0
    return max(1, int(round(len(text) / CHAR_PER_TOKEN_APPROX)))


class TokenCounter(Protocol):
    """Callable interface — ``count(text) -> tokens``."""

    def count(self, text: str) -> int: ...  # noqa: E704


@dataclass
class TiktokenCounter:
    """Tokenizer for OpenAI-family models."""
    model_id: str

    def __post_init__(self) -> None:
        try:
            import tiktoken  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "tiktoken is required for OpenAI token counting. "
                "Install: pip install tiktoken"
            ) from exc
        # Try model-specific encoding; fall back to o200k_base for new models.
        try:
            self._enc = tiktoken.encoding_for_model(self.model_id)
        except Exception:
            self._enc = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


@dataclass
class OllamaTokenCounter:
    """Ollama HTTP-based tokenizer for local + cloud models.

    Falls back to :func:`universal_estimate` if the endpoint is unavailable.
    """
    model_id: str
    host: str = "http://127.0.0.1:11434"
    timeout_s: int = 30

    def __post_init__(self) -> None:
        env_host = os.getenv("OLLAMA_HOST")
        if env_host:
            self.host = env_host.rstrip("/")

    def count(self, text: str) -> int:
        if not text:
            return 0
        try:
            r = requests.post(
                f"{self.host}/api/tokenize",
                json={"model": self.model_id, "text": text},
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            toks = data.get("tokens")
            if isinstance(toks, list):
                return len(toks)
        except Exception:
            pass
        return universal_estimate(text)


@dataclass
class UniversalCounter:
    """Universal char/4 counter — always available, never fails."""

    def count(self, text: str) -> int:
        return universal_estimate(text)


def count_tokens(text: str, *, counter: Optional[TokenCounter] = None) -> int:
    """Count tokens using ``counter`` if given, else :func:`universal_estimate`."""
    if counter is None:
        return universal_estimate(text)
    return counter.count(text)


def is_feasible(
    prompt_text: str,
    *,
    context_window: int,
    max_output_tokens: int,
    safety_margin_tokens: int = 128,
    counter: Optional[TokenCounter] = None,
) -> tuple[bool, int]:
    """Return ``(feasible, assembled_tokens)``.

    A run is feasible when::

        assembled + max_output + safety_margin <= context_window
    """
    assembled = count_tokens(prompt_text, counter=counter)
    budget = assembled + max_output_tokens + safety_margin_tokens
    return budget <= context_window, assembled
