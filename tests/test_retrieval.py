"""Tests for the Phase 4b retrieval primitives.

Covers:
- Loading and validating the 4 fixed sub-queries from experiment.yaml
- The ``search_query:`` prefix contract on every sub-query
- Domain-placeholder substitution
- The k_step splitting policy across 4 sub-queries (remainder-spread-first)

These tests do not touch Chroma or Ollama. An integration test that hits a
live index lives elsewhere (added in Phase 5).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fame.retrieval.query_templates import format_sub_query, load_sub_queries
from fame.retrieval.service import RetrievalService


REPO = Path(__file__).resolve().parents[1]


def test_experiment_yaml_defines_four_sub_queries() -> None:
    qs = load_sub_queries(REPO / "config/experiment.yaml")
    assert len(qs) == 4


def test_every_sub_query_carries_the_query_prefix() -> None:
    qs = load_sub_queries(REPO / "config/experiment.yaml")
    for q in qs:
        assert q.startswith("search_query: "), q


def test_domain_placeholder_substitution() -> None:
    t = "search_query: what {domain} approach this paper proposes"
    assert format_sub_query(t, domain="model repair") == \
           "search_query: what model repair approach this paper proposes"


def test_split_k_even() -> None:
    assert RetrievalService.split_k(8) == [2, 2, 2, 2]


def test_split_k_with_remainder_spreads_first() -> None:
    # 5 across 4 → [2, 1, 1, 1]
    assert RetrievalService.split_k(5) == [2, 1, 1, 1]
    # 7 across 4 → [2, 2, 2, 1]
    assert RetrievalService.split_k(7) == [2, 2, 2, 1]
    # 3 across 4 → [1, 1, 1, 0]  (fewer results than sub-queries)
    assert RetrievalService.split_k(3) == [1, 1, 1, 0]


def test_split_k_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        RetrievalService.split_k(0)
    with pytest.raises(ValueError):
        RetrievalService.split_k(-1)


def test_split_k_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        RetrievalService.split_k(4, remainder_policy="round_robin")
