from __future__ import annotations

from pathlib import Path

from scripts.analyse_pilot_quality import (
    _best_generated_matches,
    _citation_first_seen,
    _normalise_name,
    _parse_trace,
    _permutation_key,
)


def test_trace_parser_uses_frozen_comma_syntax() -> None:
    assert _parse_trace("Evidence sentence. Trace: [rep_01, rep_17]") == ["rep_01", "rep_17"]
    assert _parse_trace("Trace: [rep_01]; trailing") == []


def test_name_collision_keys_are_distinct_from_permutation_keys() -> None:
    assert _normalise_name("Bidirectional-Transformation") == _normalise_name("Bidirectional_Transformation")
    assert _permutation_key("A_B") == _permutation_key("B_A")
    assert _normalise_name("A_B") != _normalise_name("B_A")


def test_citation_first_seen_tracks_trace_update_not_feature_birth(tmp_path: Path) -> None:
    iterations = tmp_path / "fm_iter"
    iterations.mkdir()
    (iterations / "step_00.xml").write_text(
        "<?xml version='1.0'?><featureModel><struct><and name='Root'>"
        "<description>Root. Trace: [rep_01]</description>"
        "<feature name='A'><description>A. Trace: [rep_01]</description></feature>"
        "</and></struct><constraints/></featureModel>"
    )
    (iterations / "step_01.xml").write_text(
        "<?xml version='1.0'?><featureModel><struct><and name='Root'>"
        "<description>Root. Trace: [rep_01]</description>"
        "<feature name='A'><description>A. Trace: [rep_01, rep_02]</description></feature>"
        "</and></struct><constraints/></featureModel>"
    )
    seen = _citation_first_seen(tmp_path)
    assert seen[("A", "rep_01")] == 0
    assert seen[("A", "rep_02")] == 1


def test_reverse_match_selects_best_generated_feature(tmp_path: Path) -> None:
    path = tmp_path / "d15.csv"
    path.write_text(
        "run_id,reference_feature,generated_feature,similarity\n"
        "r1,GT_A,weak,0.41\n"
        "r1,GT_A,strong,0.82\n"
    )
    best = _best_generated_matches(path)
    assert best[("r1", "GT_A")] == {"generated_feature": "strong", "similarity": 0.82}
