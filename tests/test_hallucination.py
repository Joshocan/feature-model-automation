"""Tests for fame/evaluation/hallucination.py (L2 check)."""
from __future__ import annotations

import pytest

from fame.evaluation.hallucination import (
    HallucinationRecord,
    HallucinationReport,
    check_hallucinations,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builder
# ─────────────────────────────────────────────────────────────────────────────

def _ctx(step: int, batch: list[str], chunks: list[str] | None = None) -> dict:
    return {
        "step_index": step,
        "batch_doc_ids": batch,
        "chunk_doc_ids": chunks if chunks is not None else batch,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Happy paths
# ─────────────────────────────────────────────────────────────────────────────

def test_all_citations_in_context() -> None:
    ctx = [_ctx(0, ["d_01", "d_02"]),
           _ctx(1, ["d_03", "d_04"])]
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_01", "first_seen_step": 0},
        {"feature_id": "F1", "cited_doc_id": "d_04", "first_seen_step": 1},
    ]
    r = check_hallucinations(provenance_rows=rows, context_lines=ctx)
    assert r.n_checked == 2
    assert r.n_in_context == 2
    assert r.n_hallucinated == 0
    assert r.hallucination_rate == 0.0


def test_citation_of_doc_not_in_step_context_is_hallucination() -> None:
    """d_01 was in step 0, not step 1. Feature born at step 1 citing d_01 is L2."""
    ctx = [_ctx(0, ["d_01"]),
           _ctx(1, ["d_02"])]
    rows = [
        {"feature_id": "F1", "cited_doc_id": "d_01", "first_seen_step": 1},
    ]
    r = check_hallucinations(provenance_rows=rows, context_lines=ctx)
    assert r.n_checked == 1
    assert r.n_hallucinated == 1
    assert r.hallucination_rate == 1.0
    assert len(r.hallucinated) == 1
    off = r.hallucinated[0]
    assert isinstance(off, HallucinationRecord)
    assert off.cited_doc_id == "d_01"
    assert off.first_seen_step == 1
    assert off.was_in_context is False
    assert "d_02" in off.docs_in_context


def test_chunk_doc_ids_count_toward_in_context() -> None:
    """RAG: a doc can be in chunks_in_context even if not in the batch."""
    ctx = [_ctx(0, batch=["d_01"], chunks=["d_01", "d_02"])]
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_02", "first_seen_step": 0},
    ]
    r = check_hallucinations(provenance_rows=rows, context_lines=ctx)
    assert r.n_in_context == 1
    assert r.n_hallucinated == 0


def test_row_without_step_is_skipped() -> None:
    ctx = [_ctx(0, ["d_01"])]
    rows = [{"feature_id": "F0", "cited_doc_id": "d_01"}]      # no step
    r = check_hallucinations(provenance_rows=rows, context_lines=ctx)
    assert r.n_checked == 0
    assert r.n_skipped_no_step == 1


def test_citation_of_never_seen_doc_is_L1_not_L2() -> None:
    """Citing d_hallucinated — never in any batch — is L1 (referential), not L2."""
    ctx = [_ctx(0, ["d_01"])]
    rows = [{"feature_id": "F0", "cited_doc_id": "d_hallucinated",
             "first_seen_step": 0}]
    r = check_hallucinations(provenance_rows=rows, context_lines=ctx)
    assert r.n_checked == 0
    assert r.n_skipped_unknown_doc == 1


def test_per_step_bucketing() -> None:
    ctx = [_ctx(0, ["d_01"]), _ctx(1, ["d_02"])]
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_01", "first_seen_step": 0},
        {"feature_id": "F1", "cited_doc_id": "d_02", "first_seen_step": 1},
        {"feature_id": "F2", "cited_doc_id": "d_01", "first_seen_step": 1},   # hallucination
    ]
    r = check_hallucinations(provenance_rows=rows, context_lines=ctx)
    assert r.per_step[0]["checked"]      == 1
    assert r.per_step[0]["in_context"]   == 1
    assert r.per_step[0]["hallucinated"] == 0
    assert r.per_step[1]["checked"]      == 2
    assert r.per_step[1]["in_context"]   == 1
    assert r.per_step[1]["hallucinated"] == 1


def test_l2_applicable_flag() -> None:
    r_empty = check_hallucinations(provenance_rows=[], context_lines=[_ctx(0, ["d_01"])])
    assert r_empty.l2_applicable is False

    r_some = check_hallucinations(
        provenance_rows=[{"feature_id": "F", "cited_doc_id": "d_01", "first_seen_step": 0}],
        context_lines=[_ctx(0, ["d_01"])],
    )
    assert r_some.l2_applicable is True
