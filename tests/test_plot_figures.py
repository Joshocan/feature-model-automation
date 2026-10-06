from __future__ import annotations

from pathlib import Path

import pytest

from scripts.plot_figures import (_render_lines, prepare_family_boxplot,
                                  prepare_n_curve, prepare_tau_sweep)


def _cell(arm: str, N: int, mean: float, *, metamodel="true", order="primary") -> dict:
    return {"corpus": "repair", "model_id": "model-a", "grounding": "rag",
            "arm": arm, "N": str(N), "ordering_id": order,
            "metamodel_block": metamodel,
            "semantic__semantic_f1_total__mean": str(mean),
            "semantic__semantic_f1_total__n_ok": "5"}


def test_n_curve_excludes_ablation_order_and_sweep() -> None:
    rows = [_cell("guided_baseline", 1, .2), _cell("guided_curve", 5, .4),
            _cell("ablation", 10, .9), _cell("order_sensitivity", 10, .8),
            _cell("guided_headline", 10, .5, metamodel="false"),
            _cell("guided_headline", 10, .6, order="alt_1")]
    assert prepare_n_curve(rows) == {("repair", "model-a", "rag"):
                                     [(1, .2, 5), (5, .4, 5)]}


def test_n_curve_rejects_duplicate_guided_cells() -> None:
    with pytest.raises(ValueError, match="Multiple guided"):
        prepare_n_curve([_cell("guided_curve", 5, .4),
                         _cell("guided_headline", 5, .5)])


def test_family_boxplot_uses_declared_filters_and_population() -> None:
    family = [{"comparison": "h1", "metric": "semantic__semantic_f1_total",
               "arm_a_filter": '{"grounding":"rag","N":10}',
               "arm_b_filter": '{"grounding":"nonrag","N":10}',
               "semantic_population": "completed_extractable_final",
               "n_a": "1", "n_b": "1"}]
    wide = [
        {"grounding": "rag", "N": "10", "primary_semantic_eligible": "true",
         "semantic__semantic_f1_total": ".7", "semantic__semantic_f1_total__status": "ok"},
        {"grounding": "rag", "N": "10", "primary_semantic_eligible": "false",
         "semantic__semantic_f1_total": ".9", "semantic__semantic_f1_total__status": "ok"},
        {"grounding": "nonrag", "N": "10", "primary_semantic_eligible": "true",
         "semantic__semantic_f1_total": ".4", "semantic__semantic_f1_total__status": "ok"},
    ]
    assert prepare_family_boxplot(family, wide) == [("h1:a", [.7]), ("h1:b", [.4])]
    family[0]["n_a"] = "2"
    with pytest.raises(ValueError, match="results file says 2"):
        prepare_family_boxplot(family, wide)


def test_tau_sweep_groups_only_primary_guided_runs() -> None:
    wide = [
        {"run_id": "a", "corpus": "repair", "model_id": "m", "grounding": "rag",
         "N": "10", "arm": "guided_headline", "ordering_id": "primary",
         "metamodel_block": "true", "primary_semantic_eligible": "true"},
        {"run_id": "b", "corpus": "repair", "model_id": "m", "grounding": "rag",
         "N": "10", "arm": "guided_headline", "ordering_id": "primary",
         "metamodel_block": "true", "primary_semantic_eligible": "true"},
        {"run_id": "c", "corpus": "repair", "model_id": "m", "grounding": "rag",
         "N": "10", "arm": "ablation", "ordering_id": "primary",
         "metamodel_block": "false", "primary_semantic_eligible": "true"},
    ]
    tau = [{"run_id": rid, "tau": ".4", "f1_total": score,
            "matching_policy": "independent_max"}
           for rid, score in (("a", ".5"), ("b", ".7"), ("c", "1"))]
    tau.append({"run_id": "a", "tau": ".4", "f1_total": "0",
                "matching_policy": "one_to_one"})
    assert prepare_tau_sweep(tau, wide) == {("repair", "m", "rag", 10): [(.4, .6, 2)]}


def test_plot_renderer_writes_pdf_and_png(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    path = tmp_path / "tiny"
    _render_lines([("fixture", [1, 5], [.2, .4])], title="Fixture",
                  xlabel="N", ylabel="F1", path=path, figsize=(3, 2), dpi=72,
                  counts=[[2, 3]])
    assert path.with_suffix(".pdf").stat().st_size > 0
    assert path.with_suffix(".png").stat().st_size > 0
