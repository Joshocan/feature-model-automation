"""Recency-bias check.

If provenance citations skew toward recently-seen batches, then the extractor
is picking up positional signal rather than semantic relevance — and L0/L1/L2
provenance rates become uninterpretable. This is the *critical* sanity check
for RQ6.

Method
------
For each provenance row ``(feature, cited_doc_id, first_seen_step)``:

1. Locate the batch(es) the cited doc appeared in via D9 context_log.
2. Compute *citation offset* = ``step_of_citation - step_of_earliest_appearance``.
   * Offset 0 = feature cites the batch it was born in.
   * Offset < 0 = feature cites a batch AFTER the one in which the citation
                  was first available. (Impossible unless bookkeeping is broken.)
   * Offset > 0 = feature cites a batch earlier than it was born.
3. The distribution of offsets across all rows in a run tells us whether the
   model cites *the latest batch* disproportionately (offset near 0) or
   spreads citations across batches (varied offsets).

Skew metric
-----------
``skew`` = mean offset. A model that cites uniformly across seen batches has
skew ≈ mean(seen_span / 2). A model that cites only the last-seen batch has
skew ≈ 0. Values close to zero on high-N runs signal recency bias.

We keep :func:`recency_distribution` pure so downstream analysis code can
report percentiles, plot histograms, or aggregate across runs.
"""
from __future__ import annotations

import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class RecencyResult:
    """Aggregate recency signal for one run."""
    n_citations: int
    offsets:     List[int]
    mean_offset: float
    median_offset: float
    max_offset:  int
    zero_offset_share: float     # fraction of citations at offset 0 (born-and-cited-together)
    negative_offset_count: int   # bookkeeping-integrity check; should be 0
    per_step_citation_counts: Dict[int, int]      # step_index → count of citations FROM that step

    @property
    def is_recency_biased(self) -> bool:
        """Heuristic: if > 50% of citations sit at offset 0 AND max_offset > 0,
        the model is disproportionately citing the current batch."""
        if self.n_citations == 0:
            return False
        return self.zero_offset_share > 0.5 and self.max_offset > 0


def load_context_log(path: Path | str) -> List[dict]:
    """Read D9 ``context_log.jsonl`` line-by-line into a list of dicts."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"context_log not found: {p}")
    out: List[dict] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _first_step_per_doc(context_lines: Sequence[Mapping]) -> Dict[str, int]:
    """For each doc_id ever seen, the earliest step_index in which it appeared."""
    out: Dict[str, int] = {}
    for line in context_lines:
        step = int(line.get("step_index", -1))
        # doc_ids that made it into the prompt = batch_doc_ids (guaranteed in
        # scope) and chunk_doc_ids (retrieved chunks). We use batch_doc_ids so
        # a doc counts as "seen" from the step it was in the batch, not a
        # later step where it appeared in retrieval.
        for did in line.get("batch_doc_ids", []):
            out.setdefault(did, step)
    return out


def recency_distribution(
    *,
    provenance_rows: Iterable[Mapping],
    context_lines:   Sequence[Mapping],
) -> RecencyResult:
    """Compute recency stats for one run.

    ``provenance_rows`` — iterable of dicts with keys ``feature_id``,
    ``cited_doc_id``, ``first_seen_step``. Rows without a ``first_seen_step``
    are skipped (they belong to Non-RAG N=1 where the concept is vacuous).

    ``context_lines`` — list of dicts loaded from ``context_log.jsonl``.
    """
    first_seen = _first_step_per_doc(context_lines)
    offsets: List[int] = []
    negative = 0
    per_step: Dict[int, int] = {}

    for row in provenance_rows:
        cited = row.get("cited_doc_id")
        step  = row.get("first_seen_step")
        if cited is None or step is None:
            continue
        step = int(step)
        origin_step = first_seen.get(cited)
        if origin_step is None:
            # citation for a doc never actually in a batch — that's an L1 hallucination,
            # not a recency issue. Skip.
            continue
        offset = step - origin_step
        offsets.append(offset)
        if offset < 0:
            negative += 1
        per_step[step] = per_step.get(step, 0) + 1

    n = len(offsets)
    if n == 0:
        return RecencyResult(
            n_citations=0, offsets=[], mean_offset=0.0, median_offset=0.0,
            max_offset=0, zero_offset_share=0.0, negative_offset_count=0,
            per_step_citation_counts={},
        )

    return RecencyResult(
        n_citations=n,
        offsets=offsets,
        mean_offset=st.mean(offsets),
        median_offset=st.median(offsets),
        max_offset=max(offsets) if offsets else 0,
        zero_offset_share=sum(1 for o in offsets if o == 0) / n,
        negative_offset_count=negative,
        per_step_citation_counts=per_step,
    )
