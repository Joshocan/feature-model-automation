from __future__ import annotations

from typing import Dict, Optional


DEFAULT_MODEL_WINDOWS: Dict[str, int] = {
    "gpt-oss:120b-cloud": 131_000,
    "glm-4.7:cloud": 200_000,
    "deepseek-v3.2:cloud": 128_000,
}


def estimate_max_chars_from_tokens(tokens: int, *, chars_per_token: float = 4.0, safety: float = 0.8) -> int:
    if tokens <= 0:
        return 0
    return int(tokens * chars_per_token * safety)


def compute_max_total_chars(
    model: str,
    *,
    model_windows: Optional[Dict[str, int]] = None,
    chars_per_token: float = 4.0,
    safety: float = 0.8,
) -> int:
    windows = model_windows or DEFAULT_MODEL_WINDOWS
    tokens = windows.get(model, 0)
    return estimate_max_chars_from_tokens(tokens, chars_per_token=chars_per_token, safety=safety)


def compute_max_chunks(
    *,
    max_total_chars: int,
    baseline_chars: int = 140_000,
    baseline_chunks: int = 120,
) -> int:
    if max_total_chars <= 0:
        return baseline_chunks
    ratio = max_total_chars / baseline_chars
    est = int(baseline_chunks * ratio)
    return max(1, est)
