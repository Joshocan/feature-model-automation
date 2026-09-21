"""Deterministic batching (Phase 5.1).

Split a frozen ordering π into exactly N non-empty contiguous slices.

Policy: **floor with remainder spread first.** Base size = floor(n / N). The
first ``n mod N`` batches take one extra doc. This matches the brief's "docs
per step" table (§1) and keeps every batch non-empty for every valid N.

  N=5,  n=54 → [11, 11, 11, 11, 10]     (floor=10, rem=4)
  N=10, n=54 → [ 6,  6,  6,  6,  5,  5,  5,  5,  5,  5]  (floor=5, rem=4)
  N=20, n=54 → [3]*14 + [2]*6           (floor=2, rem=14)
  N=54, n=54 → [1] * 54                  (floor=1, rem=0)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Batch:
    """One batch B_j of an N-slice."""
    step_index: int             # 0-based index within the campaign run
    doc_ids: List[str]          # doc_ids in π order


def slice_ordering(ordering: Sequence[str], N: int) -> List[Batch]:
    """Partition ``ordering`` into exactly N non-empty contiguous slices.

    See module docstring for the size policy.
    """
    n = len(ordering)
    if n == 0:
        raise ValueError("ordering must be non-empty")
    if N < 1 or N > n:
        raise ValueError(f"N must satisfy 1 <= N <= {n}, got {N}")

    base, rem = divmod(n, N)
    batches: List[Batch] = []
    cursor = 0
    for j in range(N):
        size = base + (1 if j < rem else 0)
        # Invariant: size >= 1 whenever N <= n (base >= 1 or rem covers early batches).
        end = cursor + size
        batches.append(Batch(step_index=j, doc_ids=list(ordering[cursor:end])))
        cursor = end
    return batches


def assert_covers_pi(batches: Sequence[Batch], ordering: Sequence[str]) -> None:
    """Sanity check — every doc appears exactly once, in π order."""
    seen: List[str] = []
    for b in batches:
        seen.extend(b.doc_ids)
    if seen != list(ordering):
        raise AssertionError(
            "batching did not cover π exactly once in order:"
            f"  expected {list(ordering)}\n"
            f"  got      {seen}"
        )
