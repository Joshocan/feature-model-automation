"""Static tests for the frozen two-lane campaign matrix."""
from __future__ import annotations

from scripts.build_run_matrix import build_matrix


def test_combined_enabled_matrix_acceptance_totals() -> None:
    matrix = build_matrix("ifs-2027", only_enabled=True, lane="all")
    assert matrix["total_runs"] == 694
    assert matrix["total_calls"] == 6800
    assert matrix["per_lane_runs"] == {"open_weight": 644, "astra": 50}
    assert matrix["per_lane_calls"] == {"open_weight": 6300, "astra": 500}


def test_astra_reduced_cells_are_exact() -> None:
    matrix = build_matrix("ifs-2027", only_enabled=True, lane="astra")
    assert matrix["total_runs"] == 50
    assert matrix["total_calls"] == 500
    signatures = {
        (r["extra"]["arm"], r["corpus"], r["grounding"],
         r["metamodel_block"], r["N"])
        for r in matrix["runs"]
    }
    assert signatures == {
        ("guided_headline", "repair", "rag", True, 10),
        ("guided_headline", "repair", "nonrag", True, 10),
        ("ablation", "repair", "rag", False, 10),
        ("ablation", "federation", "rag", False, 10),
        ("astra_cross_corpus", "federation", "rag", True, 10),
    }


def test_open_weight_matrix_and_sweep_seeds() -> None:
    matrix = build_matrix("ifs-2027", only_enabled=True, lane="open_weight")
    assert matrix["total_runs"] == 644
    assert matrix["total_calls"] == 6300
    assert set(matrix["per_model_calls"].values()) == {3150}
    sweep_seeds = {
        r["seed"] for r in matrix["runs"]
        if r["extra"]["arm"].startswith("k_doc_sweep_")
    }
    assert sweep_seeds == {20, 21, 22}
