"""Tests for fame/evaluation/reachable.py (RQ2 calibration)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import pytest

from fame.evaluation.reachable import (
    DualRecall,
    FeaturePartition,
    dual_recall,
    load_attribution,
    load_partition,
    reach_features,
    rho,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _write_attribution(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gt_feature_id", "doc_id", "source"])
        for f, d, s in rows:
            w.writerow([f, d, s])


def _write_partition(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gt_feature_id", "partition"])
        for f, k in rows:
            w.writerow([f, k])


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def test_load_attribution_groups_docs_per_feature(tmp_path: Path) -> None:
    p = tmp_path / "attr.csv"
    _write_attribution(p, [
        ("Repair",  "rep_01", "src"),
        ("Repair",  "rep_02", "src"),
        ("Rollback", "rep_03", "src"),
    ])
    attr = load_attribution(p)
    assert attr["Repair"]  == {"rep_01", "rep_02"}
    assert attr["Rollback"] == {"rep_03"}


def test_load_attribution_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_attribution(tmp_path / "nope.csv")


def test_load_partition_splits_attested_and_organising(tmp_path: Path) -> None:
    p = tmp_path / "part.csv"
    _write_partition(p, [
        ("Repair",     "attested"),
        ("Rollback",   "attested"),
        ("Structural", "organising"),
        ("Facet_Root", "organising"),
    ])
    part = load_partition(p)
    assert part.attested   == {"Repair", "Rollback"}
    assert part.organising == {"Structural", "Facet_Root"}
    assert part.full       == {"Repair", "Rollback", "Structural", "Facet_Root"}


def test_load_partition_rejects_unknown_kind(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    _write_partition(p, [("X", "wibble")])
    with pytest.raises(ValueError):
        load_partition(p)


# ─────────────────────────────────────────────────────────────────────────────
# Reach subset
# ─────────────────────────────────────────────────────────────────────────────

def test_reach_features_empty_corpus_yields_empty() -> None:
    attr = {"F1": {"d1", "d2"}}
    assert reach_features(attribution=attr, corpus_doc_ids=[], parents={'F1': None}) == frozenset()


def test_reach_features_full_corpus_yields_all_attested() -> None:
    attr = {"F1": {"d1"}, "F2": {"d2"}, "F3": {"d1", "d2"}}
    reach = reach_features(attribution=attr, corpus_doc_ids=["d1", "d2"], parents={f: None for f in attr})
    assert reach == {"F1", "F2", "F3"}


def test_reach_features_subset_corpus_reaches_subset() -> None:
    attr = {"F1": {"d1"}, "F2": {"d2"}, "F3": {"d3"}}
    reach = reach_features(attribution=attr, corpus_doc_ids=["d1", "d2"], parents={f: None for f in attr})
    assert reach == {"F1", "F2"}


def test_reach_features_only_attested_filters_organising() -> None:
    attr = {"F1": {"d1"}, "F2": {"d2"}, "F3_org": {"d1"}}   # stray row
    reach = reach_features(
        attribution=attr,
        corpus_doc_ids=["d1", "d2"],
        only_attested={"F1", "F2"},
        parents={f: None for f in attr},
    )
    assert reach == {"F1", "F2"}         # F3_org excluded by attested filter


def test_upward_closure_adds_unattributed_ancestors_not_siblings():
    parents = {'Root': None, 'Group': 'Root', 'Leaf': 'Group', 'Sibling': 'Root'}
    result = reach_features(attribution={'Leaf': {'d1'}}, corpus_doc_ids=['d1'],
                            parents=parents, only_attested={'Leaf'})
    assert result == {'Leaf', 'Group', 'Root'}


@pytest.mark.parametrize('parents', [{'A': 'missing'}, {'A': 'B', 'B': 'A'}])
def test_bad_parent_graph_rejected(parents):
    with pytest.raises(ValueError):
        reach_features(attribution={'A': {'d'}}, corpus_doc_ids=['d'], parents=parents)


# ─────────────────────────────────────────────────────────────────────────────
# rho
# ─────────────────────────────────────────────────────────────────────────────

def test_rho_basic() -> None:
    assert rho(36, 130) == pytest.approx(36 / 130)


def test_rho_full_corpus_at_most_one() -> None:
    assert rho(100, 100) == 1.0
    assert rho(0, 100) == 0.0


def test_rho_rejects_empty_F_t() -> None:
    with pytest.raises(ValueError):
        rho(1, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Dual recall
# ─────────────────────────────────────────────────────────────────────────────

def test_dual_recall_perfect_extraction_on_reach() -> None:
    part = FeaturePartition(
        attested=frozenset({"A", "B", "C"}),
        organising=frozenset({"Root", "Facet"}),
    )
    reach = {"A", "B", "C"}
    r = dual_recall(
        corpus="test",
        matched_features={"A", "B", "C"},
        partition=part,
        reach=reach,
    )
    assert r.recall_vs_reach   == 1.0
    assert r.recall_vs_F_t     == 3 / 5
    assert r.recall_vs_F_t_att == 1.0
    assert r.recall_vs_F_t_org == 0.0
    assert r.n_extracted_matched == 3


def test_dual_recall_organising_only_gives_low_att_recall() -> None:
    part = FeaturePartition(
        attested=frozenset({"A", "B"}),
        organising=frozenset({"Root"}),
    )
    reach = {"A", "B"}
    r = dual_recall(
        corpus="test",
        matched_features={"Root"},          # only organising matched
        partition=part,
        reach=reach,
    )
    assert r.recall_vs_reach   == 0.0
    assert r.recall_vs_F_t_att == 0.0
    assert r.recall_vs_F_t_org == 1.0
    assert r.recall_vs_F_t     == 1 / 3


def test_dual_recall_empty_reach_returns_zero_not_error() -> None:
    part = FeaturePartition(
        attested=frozenset({"A"}),
        organising=frozenset(),
    )
    r = dual_recall(
        corpus="test",
        matched_features=set(),
        partition=part,
        reach=set(),        # empty reach — corpus subset didn't cover A
    )
    assert r.recall_vs_reach == 0.0
    assert r.reach_size == 0


def test_dual_recall_ignores_extracted_features_not_in_F_t() -> None:
    """Runs may match hallucinated features not in F_t. They don't count."""
    part = FeaturePartition(
        attested=frozenset({"A"}),
        organising=frozenset(),
    )
    r = dual_recall(
        corpus="test",
        matched_features={"A", "Hallucinated_X"},
        partition=part,
        reach={"A"},
    )
    assert r.recall_vs_reach == 1.0
    assert r.n_extracted_matched == 1     # only A counted
