from __future__ import annotations

import json
from pathlib import Path

import pytest

from fame.evaluation.provenance import (
    PROVENANCE_METRICS,
    compute_first_seen_pairs,
    evaluate_provenance,
    extract_feature_descriptions,
)


FM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<featureModel>
  <struct>
    <and mandatory="true" name="Root">
      {children}
    </and>
  </struct>
</featureModel>
"""


def _feature(name: str, description: str | None = None) -> str:
    if description is None:
        return f'<feature name="{name}"/>'
    return f'<feature name="{name}"><description>{description}</description></feature>'


def _write_fm(path: Path, features: list[tuple[str, str | None]]) -> Path:
    children = "\n      ".join(_feature(n, d) for n, d in features)
    path.write_text(FM_TEMPLATE.format(children=children), encoding="utf-8")
    return path


def _write_context_log(path: Path, entries: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# extract_feature_descriptions
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_feature_descriptions_returns_root_and_children(tmp_path: Path) -> None:
    fm = _write_fm(tmp_path / "fm.xml", [
        ("Alpha", "Something. Trace: [rep_01]"),
        ("Beta", None),
    ])
    feats = extract_feature_descriptions(fm)
    assert [f["name"] for f in feats] == ["Root", "Alpha", "Beta"]
    assert feats[1]["description"].endswith("Trace: [rep_01]")
    assert feats[2]["description"] is None


def test_extract_feature_descriptions_no_struct(tmp_path: Path) -> None:
    (tmp_path / "fm.xml").write_text(
        '<?xml version="1.0"?><featureModel/>', encoding="utf-8")
    assert extract_feature_descriptions(tmp_path / "fm.xml") == []


# ─────────────────────────────────────────────────────────────────────────────
# compute_first_seen_pairs
# ─────────────────────────────────────────────────────────────────────────────

def test_first_seen_pairs_records_earliest_step(tmp_path: Path) -> None:
    iter_dir = tmp_path / "fm_iter"
    iter_dir.mkdir()
    _write_fm(iter_dir / "step_00.xml", [("Alpha", "Trace: [rep_01]")])
    _write_fm(iter_dir / "step_01.xml", [
        ("Alpha", "Trace: [rep_01, rep_02]"),
        ("Beta",  "Trace: [rep_02]"),
    ])
    _write_fm(iter_dir / "step_02.xml", [
        ("Alpha", "Trace: [rep_01, rep_02]"),
        ("Beta",  "Trace: [rep_02, rep_03]"),
    ])
    pairs = compute_first_seen_pairs(iter_dir)
    expected = [
        {"feature_id": "Alpha", "cited_doc_id": "rep_01", "first_seen_step": 0},
        {"feature_id": "Alpha", "cited_doc_id": "rep_02", "first_seen_step": 1},
        {"feature_id": "Beta",  "cited_doc_id": "rep_02", "first_seen_step": 1},
        {"feature_id": "Beta",  "cited_doc_id": "rep_03", "first_seen_step": 2},
    ]
    assert pairs == expected


def test_first_seen_pairs_skips_unparseable_markers(tmp_path: Path) -> None:
    iter_dir = tmp_path / "fm_iter"
    iter_dir.mkdir()
    _write_fm(iter_dir / "step_00.xml", [
        ("Alpha", "no marker here"),
        ("Beta", "[not-a-trace]"),
    ])
    assert compute_first_seen_pairs(iter_dir) == []


def test_first_seen_pairs_empty_dir(tmp_path: Path) -> None:
    assert compute_first_seen_pairs(tmp_path / "does_not_exist") == []


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_provenance — envelope form
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_provenance_missing_gen_is_missing_artifact(tmp_path: Path) -> None:
    result = evaluate_provenance(
        tmp_path / "gen.xml", tmp_path / "iter", tmp_path / "log.jsonl",
        known_doc_ids=["rep_01"],
    )
    assert all(env["status"] == "missing_artifact"
               for env in result["metrics"].values())
    assert set(result["metrics"]) == set(PROVENANCE_METRICS)


def test_evaluate_provenance_no_features_is_ineligible(tmp_path: Path) -> None:
    (tmp_path / "fm.xml").write_text(
        '<?xml version="1.0"?><featureModel><struct/></featureModel>',
        encoding="utf-8")
    result = evaluate_provenance(
        tmp_path / "fm.xml", tmp_path / "iter", tmp_path / "log.jsonl",
    )
    m = result["metrics"]
    assert m["n_features"]["value"] == 0
    assert m["feature_trace_coverage"]["status"] == "ineligible"


def test_evaluate_provenance_all_l0_l1_ok_no_context(tmp_path: Path) -> None:
    """L0 + L1 computable from fm_gen alone; L2/recency lack context → missing_artifact."""
    fm = _write_fm(tmp_path / "fm.xml", [
        ("Alpha", "Something. Trace: [rep_01, rep_02]"),
        ("Beta",  "Something else. Trace: [rep_02]"),
    ])
    result = evaluate_provenance(
        fm, tmp_path / "iter_missing", tmp_path / "log_missing.jsonl",
        known_doc_ids=["rep_01", "rep_02", "rep_03"],
    )
    m = result["metrics"]
    # Root has no description → 2 of 3 features carry parseable traces.
    assert m["n_features"]["value"] == 3
    assert m["n_features_with_parseable_trace"]["value"] == 2
    assert m["feature_trace_coverage"]["value"] == pytest.approx(2 / 3)
    assert m["L0_marker_emission_compliance"]["value"] == pytest.approx(2 / 3)
    assert m["L0_parse_rate"]["value"] == 1.0
    assert m["n_citation_pairs"]["value"] == 3
    assert m["n_unique_cited_doc_ids"]["value"] == 2
    assert m["L1_referential_integrity"]["value"] == 1.0
    assert m["L2_hallucination_rate"]["status"] == "missing_artifact"
    assert m["recency_mean_offset"]["status"] == "missing_artifact"


def test_evaluate_provenance_l1_flags_unknown_doc(tmp_path: Path) -> None:
    fm = _write_fm(tmp_path / "fm.xml", [
        ("Alpha", "x. Trace: [rep_01, made_up]"),
    ])
    result = evaluate_provenance(
        fm, tmp_path / "iter", tmp_path / "log.jsonl",
        known_doc_ids=["rep_01"],
    )
    m = result["metrics"]
    # 1 of 2 pairs pass L1 (rep_01 known, made_up unknown).
    assert m["L1_referential_integrity"]["value"] == 0.5
    assert m["L1_referential_integrity"]["numerator"] == 1
    assert m["L1_referential_integrity"]["denominator"] == 2


def test_evaluate_provenance_l1_unsupported_when_no_known_set(tmp_path: Path) -> None:
    fm = _write_fm(tmp_path / "fm.xml", [("Alpha", "x. Trace: [rep_01]")])
    result = evaluate_provenance(
        fm, tmp_path / "iter", tmp_path / "log.jsonl",
        known_doc_ids=[],
    )
    assert result["metrics"]["L1_referential_integrity"]["status"] == "unsupported"


def test_evaluate_provenance_l0_parse_rate_undefined_when_no_markers(tmp_path: Path) -> None:
    fm = _write_fm(tmp_path / "fm.xml", [("Alpha", "no marker")])
    result = evaluate_provenance(
        fm, tmp_path / "iter", tmp_path / "log.jsonl", known_doc_ids=["rep_01"],
    )
    m = result["metrics"]
    assert m["L0_marker_emission_compliance"]["value"] == 0.0
    assert m["L0_parse_rate"]["status"] == "not_applicable"
    assert m["L1_referential_integrity"]["status"] == "not_applicable"


def test_evaluate_provenance_l2_detects_hallucination(tmp_path: Path) -> None:
    fm = _write_fm(tmp_path / "fm.xml", [
        ("Alpha", "x. Trace: [rep_01]"),
        ("Beta",  "y. Trace: [rep_02]"),
    ])
    iter_dir = tmp_path / "fm_iter"
    iter_dir.mkdir()
    _write_fm(iter_dir / "step_00.xml", [
        ("Alpha", "x. Trace: [rep_01]"),
    ])
    _write_fm(iter_dir / "step_01.xml", [
        ("Alpha", "x. Trace: [rep_01]"),
        ("Beta",  "y. Trace: [rep_02]"),
    ])
    # Batches: step 0 saw rep_01; step 1 saw rep_02.
    # Alpha correctly cites rep_01 at step 0.
    # Beta cites rep_02 at step 1 — also in context. No hallucination.
    log = _write_context_log(tmp_path / "log.jsonl", [
        {"step_index": 0, "batch_doc_ids": ["rep_01"], "chunk_doc_ids": []},
        {"step_index": 1, "batch_doc_ids": ["rep_02"], "chunk_doc_ids": []},
    ])
    r_ok = evaluate_provenance(fm, iter_dir, log,
                                known_doc_ids=["rep_01", "rep_02"],
                                n_planned_steps=2)
    assert r_ok["metrics"]["L2_hallucination_rate"]["value"] == 0.0
    assert r_ok["metrics"]["L2_applicable"]["value"] is True

    # Now swap contexts so Beta cites a doc it never saw at step 1.
    log_bad = _write_context_log(tmp_path / "log_bad.jsonl", [
        {"step_index": 0, "batch_doc_ids": ["rep_01"], "chunk_doc_ids": []},
        {"step_index": 1, "batch_doc_ids": ["rep_03"], "chunk_doc_ids": []},
    ])
    r_hallu = evaluate_provenance(fm, iter_dir, log_bad,
                                    known_doc_ids=["rep_01", "rep_02", "rep_03"],
                                    n_planned_steps=2)
    # rep_02 never appeared in any batch → skipped by hallucination check (L1),
    # not counted as an L2 hallucination.
    m = r_hallu["metrics"]
    assert m["L2_n_checked"]["value"] == 1     # only rep_01 was ever in context
    assert m["L2_hallucination_rate"]["value"] == 0.0


def test_evaluate_provenance_l2_vacuous_for_n1(tmp_path: Path) -> None:
    fm = _write_fm(tmp_path / "fm.xml", [
        ("Alpha", "x. Trace: [rep_01]"),
        ("Beta",  "y. Trace: [rep_02]"),
    ])
    iter_dir = tmp_path / "fm_iter"
    iter_dir.mkdir()
    _write_fm(iter_dir / "step_00.xml", [
        ("Alpha", "x. Trace: [rep_01]"),
        ("Beta",  "y. Trace: [rep_02]"),
    ])
    # N=1: single step, batch contains every corpus doc.
    log = _write_context_log(tmp_path / "log.jsonl", [
        {"step_index": 0, "batch_doc_ids": ["rep_01", "rep_02"], "chunk_doc_ids": []},
    ])
    result = evaluate_provenance(fm, iter_dir, log,
                                  known_doc_ids=["rep_01", "rep_02"],
                                  n_planned_steps=1)
    m = result["metrics"]
    assert m["L2_hallucination_rate"]["status"] == "not_applicable"
    assert m["L2_applicable"]["value"] is False


def test_evaluate_provenance_recency_flags_bias(tmp_path: Path) -> None:
    fm = _write_fm(tmp_path / "fm.xml", [
        ("Alpha", "x. Trace: [rep_01]"),
        ("Beta",  "y. Trace: [rep_02]"),
        ("Gamma", "z. Trace: [rep_03]"),
        ("Delta", "w. Trace: [rep_01]"),
    ])
    iter_dir = tmp_path / "fm_iter"
    iter_dir.mkdir()
    # Alpha, Beta, Gamma each appear at their own birth step (offset 0).
    # Delta cites rep_01 at step 2 (offset = 2-0 = 2) — proves max_offset > 0
    # so the "was another batch available?" branch of the heuristic passes.
    _write_fm(iter_dir / "step_00.xml", [("Alpha", "x. Trace: [rep_01]")])
    _write_fm(iter_dir / "step_01.xml", [
        ("Alpha", "x. Trace: [rep_01]"),
        ("Beta",  "y. Trace: [rep_02]"),
    ])
    _write_fm(iter_dir / "step_02.xml", [
        ("Alpha", "x. Trace: [rep_01]"),
        ("Beta",  "y. Trace: [rep_02]"),
        ("Gamma", "z. Trace: [rep_03]"),
        ("Delta", "w. Trace: [rep_01]"),
    ])
    log = _write_context_log(tmp_path / "log.jsonl", [
        {"step_index": 0, "batch_doc_ids": ["rep_01"], "chunk_doc_ids": []},
        {"step_index": 1, "batch_doc_ids": ["rep_02"], "chunk_doc_ids": []},
        {"step_index": 2, "batch_doc_ids": ["rep_03"], "chunk_doc_ids": []},
    ])
    result = evaluate_provenance(fm, iter_dir, log,
                                  known_doc_ids=["rep_01", "rep_02", "rep_03"],
                                  n_planned_steps=3)
    m = result["metrics"]
    # 3 of 4 citations at offset 0 → 0.75; max_offset = 2 → biased.
    assert m["recency_zero_offset_share"]["value"] == pytest.approx(3 / 4)
    assert m["recency_max_offset"]["value"] == 2
    assert m["is_recency_biased"]["value"] is True
    assert m["recency_negative_offset_count"]["value"] == 0
