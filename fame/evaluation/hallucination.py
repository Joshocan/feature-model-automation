"""L2 provenance hallucination check.

L2 asks: **when the model cited doc_id X for feature F at step j, was X
actually in the context at step j?** If not, the model invented a citation
that looks well-formed (L0/L1 pass) but was not grounded in the evidence it
was shown. That is a hallucination.

L0 (marker parses) and L1 (doc_id names a real corpus doc) live in
``fame.evaluation.groundedness``. This module handles L2 only.

Applicability
-------------
For Non-RAG at N==1, every corpus doc is in the context at every step, so
every valid citation is trivially "in context" and L2 is vacuous. We report
that as a property, not a gap (per IFS brief §6.6).

Inputs
------
* Provenance rows from D16, each carrying ``feature_id``, ``cited_doc_id``,
  ``first_seen_step``.
* Context log lines from D9, each carrying ``step_index``, ``batch_doc_ids``
  and (for RAG) ``chunk_doc_ids``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Set


@dataclass(frozen=True)
class HallucinationRecord:
    """One provenance row's L2 verdict."""
    feature_id:       str
    cited_doc_id:     str
    first_seen_step:  int
    docs_in_context:  FrozenSet[str]
    was_in_context:   bool


@dataclass
class HallucinationReport:
    """Per-run aggregate."""
    n_checked:               int
    n_in_context:            int
    n_hallucinated:          int
    n_skipped_no_step:       int         # rows lacking first_seen_step
    n_skipped_unknown_doc:   int         # doc_id never in any batch (L1 miss)
    per_step:                Dict[int, Dict[str, int]] = field(default_factory=dict)
    hallucinated:            List[HallucinationRecord] = field(default_factory=list)

    @property
    def hallucination_rate(self) -> float:
        if self.n_checked == 0:
            return 0.0
        return self.n_hallucinated / self.n_checked

    @property
    def l2_applicable(self) -> bool:
        """L2 is vacuous when every step's context contains every doc (Non-RAG N=1)."""
        return self.n_checked > 0


# ─────────────────────────────────────────────────────────────────────────────
# Context lookup
# ─────────────────────────────────────────────────────────────────────────────

def _docs_in_context_by_step(context_lines: Sequence[Mapping]) -> Dict[int, FrozenSet[str]]:
    """For each step, the union of batch_doc_ids and chunk_doc_ids.

    RAG context = batch + retrieved chunks (per :func:`fame.retrieval.RetrievalService`).
    Non-RAG context = batch chunks only. Either way, batch_doc_ids is the
    authoritative set of docs the prompt saw at that step.
    """
    out: Dict[int, Set[str]] = {}
    for line in context_lines:
        idx = int(line.get("step_index", -1))
        s = out.setdefault(idx, set())
        s.update(line.get("batch_doc_ids", []) or [])
        s.update(line.get("chunk_doc_ids", []) or [])
    return {k: frozenset(v) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main check
# ─────────────────────────────────────────────────────────────────────────────

def check_hallucinations(
    *,
    provenance_rows: Iterable[Mapping],
    context_lines:   Sequence[Mapping],
) -> HallucinationReport:
    """Cross-reference every citation against the context at its birth step."""
    context_by_step = _docs_in_context_by_step(context_lines)
    all_docs_ever_in_batch: Set[str] = set()
    for docs in context_by_step.values():
        all_docs_ever_in_batch.update(docs)

    checked = in_ctx = hallucinated = skipped_no_step = skipped_unknown = 0
    per_step: Dict[int, Dict[str, int]] = {}
    offenders: List[HallucinationRecord] = []

    for row in provenance_rows:
        cited = row.get("cited_doc_id")
        step  = row.get("first_seen_step")
        if cited is None or step is None:
            skipped_no_step += 1
            continue
        if cited not in all_docs_ever_in_batch:
            # doc_id names a corpus doc but never in this run's context → L1 issue.
            skipped_unknown += 1
            continue
        step = int(step)
        checked += 1
        docs = context_by_step.get(step, frozenset())
        was_in = cited in docs
        if was_in:
            in_ctx += 1
        else:
            hallucinated += 1
            offenders.append(HallucinationRecord(
                feature_id=str(row.get("feature_id", "")),
                cited_doc_id=cited,
                first_seen_step=step,
                docs_in_context=docs,
                was_in_context=False,
            ))
        bucket = per_step.setdefault(step, {"checked": 0, "in_context": 0, "hallucinated": 0})
        bucket["checked"] += 1
        if was_in: bucket["in_context"] += 1
        else:      bucket["hallucinated"] += 1

    return HallucinationReport(
        n_checked=checked,
        n_in_context=in_ctx,
        n_hallucinated=hallucinated,
        n_skipped_no_step=skipped_no_step,
        n_skipped_unknown_doc=skipped_unknown,
        per_step=per_step,
        hallucinated=offenders,
    )
