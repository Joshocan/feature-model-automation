from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fame.evaluation.alignment import evaluate_alignment, extract_indexed_nodes


def _write(path: Path, children: str) -> Path:
    path.write_text(
        '<featureModel><struct><and name="Root">' + children +
        '</and></struct></featureModel>', encoding="utf-8")
    return path


def test_parent_denominator_excludes_root_and_unmatched_parent(tmp_path: Path) -> None:
    ref = extract_indexed_nodes(_write(tmp_path / "ref.xml",
        '<and name="Mechanism"><feature name="Subtype"/></and>'
        '<feature name="Other"/>'))
    gen = extract_indexed_nodes(_write(tmp_path / "gen.xml",
        '<and name="Mechanism"><feature name="Subtype"/></and>'
        '<feature name="Other"/>'))
    score = evaluate_alignment(np.eye(4), gen, ref, tau=0.4)
    assert score["n_parent_evaluable"] == 3
    assert score["n_parent_correct"] == 3
    assert score["parent_match_rate"] == 1.0
    assert score["n_exact_parent_evaluable"] == 3


def test_wrong_parent_is_counted_and_attribution_is_not_blanket(tmp_path: Path) -> None:
    ref = extract_indexed_nodes(_write(tmp_path / "ref.xml",
        '<and name="A"><feature name="X"/></and><feature name="B"/>'))
    gen = extract_indexed_nodes(_write(tmp_path / "gen.xml",
        '<and name="A"/><and name="B"><feature name="X">'
        '<description>Specific. Trace: [rep_01, rep_02]</description>'
        '</feature></and>'))
    # Root/A/X/B versus Root/A/B/X.
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    score = evaluate_alignment(matrix, gen, ref, tau=0.4,
        attribution={"X": {"rep_01"}})
    assert score["n_parent_evaluable"] == 3
    assert score["n_parent_correct"] == 2
    assert score["parent_match_rate"] == pytest.approx(2 / 3)
    assert score["n_citations"] == 2
    assert score["n_attribution_applicable"] == 2
    assert score["n_attribution_agreeing"] == 1
    assert score["attribution_agreement_rate"] == 0.5


def test_exact_and_near_duplicates_separated(tmp_path: Path) -> None:
    ref = extract_indexed_nodes(_write(tmp_path / "ref.xml", '<feature name="A"/>'))
    gen = extract_indexed_nodes(_write(tmp_path / "gen.xml",
        '<feature name="A"/><feature name="A"/><feature name="A_variant"/>'))
    score = evaluate_alignment(np.ones((4, 2)), gen, ref, tau=0.4,
        generated_self_similarity=np.array([
            [1, 0, 0, 0], [0, 1, 1, .85], [0, 1, 1, .95], [0, .85, .95, 1]
        ]))
    assert score["exact_duplicate_surplus"] == 1
    assert score["near_duplicate_pairs_tau_0_8"] == 2
    assert score["near_duplicate_pairs_tau_0_9"] == 1
