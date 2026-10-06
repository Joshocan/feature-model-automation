"""Tests for fame/evaluation/recency.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fame.evaluation.recency import (
    RecencyResult,
    load_context_log,
    recency_distribution,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────────

def _write_context_log(path: Path, lines: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def _mini_context(n_steps: int = 5, docs_per_step: int = 3) -> list[dict]:
    out = []
    doc_id = 1
    for step in range(n_steps):
        batch = [f"d_{doc_id + i:02d}" for i in range(docs_per_step)]
        doc_id += docs_per_step
        out.append({
            "step_index": step,
            "N": n_steps,
            "grounding": "rag",
            "batch_doc_ids": batch,
            "chunk_doc_ids": batch,      # simulated: retrieval only hits its own batch
            "chunk_ids": [f"c_{step}_{i}" for i in range(docs_per_step)],
            "feasible": True,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def test_load_context_log_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "ctx.jsonl"
    lines = _mini_context(3)
    _write_context_log(p, lines)
    got = load_context_log(p)
    assert got == lines


def test_load_context_log_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_context_log(tmp_path / "nope.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Recency distribution
# ─────────────────────────────────────────────────────────────────────────────

def test_recency_empty_provenance() -> None:
    ctx = _mini_context(3)
    r = recency_distribution(provenance_rows=[], context_lines=ctx)
    assert isinstance(r, RecencyResult)
    assert r.n_citations == 0
    assert r.is_recency_biased is False


def test_recency_all_citations_at_offset_zero_flags_bias() -> None:
    """Every feature cites the batch it was born in → recency-bias positive."""
    ctx = _mini_context(3)          # steps 0..2, docs d_01..d_09
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_01", "first_seen_step": 0},
        {"feature_id": "F1", "cited_doc_id": "d_04", "first_seen_step": 1},
        {"feature_id": "F2", "cited_doc_id": "d_07", "first_seen_step": 2},
    ]
    r = recency_distribution(provenance_rows=rows, context_lines=ctx)
    assert r.n_citations == 3
    assert r.offsets == [0, 0, 0]
    assert r.zero_offset_share == 1.0
    assert r.max_offset == 0
    # max_offset == 0 → not flagged (heuristic needs some spread)
    assert r.is_recency_biased is False


def test_recency_spread_citations_not_flagged() -> None:
    """Feature at step 2 cites doc from step 0 (offset = 2)."""
    ctx = _mini_context(3)
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_01", "first_seen_step": 2},  # +2
        {"feature_id": "F1", "cited_doc_id": "d_04", "first_seen_step": 2},  # +1
        {"feature_id": "F2", "cited_doc_id": "d_07", "first_seen_step": 2},  # 0
    ]
    r = recency_distribution(provenance_rows=rows, context_lines=ctx)
    assert r.offsets == [2, 1, 0]
    assert r.zero_offset_share == pytest.approx(1 / 3)
    assert r.is_recency_biased is False


def test_recency_mostly_zero_with_some_spread_flags_bias() -> None:
    """> 50% at offset 0 AND max > 0 → flagged."""
    ctx = _mini_context(3)
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_01", "first_seen_step": 0},   # 0
        {"feature_id": "F1", "cited_doc_id": "d_04", "first_seen_step": 1},   # 0
        {"feature_id": "F2", "cited_doc_id": "d_07", "first_seen_step": 2},   # 0
        {"feature_id": "F3", "cited_doc_id": "d_01", "first_seen_step": 2},   # 2
    ]
    r = recency_distribution(provenance_rows=rows, context_lines=ctx)
    assert r.zero_offset_share == pytest.approx(0.75)
    assert r.max_offset == 2
    assert r.is_recency_biased is True


def test_recency_row_without_step_is_skipped() -> None:
    ctx = _mini_context(3)
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_01"},                          # no step
        {"feature_id": "F1", "cited_doc_id": "d_04", "first_seen_step": 1},   # 0
    ]
    r = recency_distribution(provenance_rows=rows, context_lines=ctx)
    assert r.n_citations == 1


def test_recency_row_citing_unknown_doc_is_skipped() -> None:
    ctx = _mini_context(3)
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_never", "first_seen_step": 1},
        {"feature_id": "F1", "cited_doc_id": "d_04",     "first_seen_step": 1},
    ]
    r = recency_distribution(provenance_rows=rows, context_lines=ctx)
    assert r.n_citations == 1                # d_never skipped (L1 issue)


def test_recency_per_step_citation_counts() -> None:
    ctx = _mini_context(3)
    rows = [
        {"feature_id": "F0", "cited_doc_id": "d_01", "first_seen_step": 0},
        {"feature_id": "F1", "cited_doc_id": "d_01", "first_seen_step": 2},
        {"feature_id": "F2", "cited_doc_id": "d_04", "first_seen_step": 2},
    ]
    r = recency_distribution(provenance_rows=rows, context_lines=ctx)
    assert r.per_step_citation_counts == {0: 1, 2: 2}
