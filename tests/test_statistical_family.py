from __future__ import annotations

import pytest

from fame.evaluation.statistical_family import Comparison, run_family


def _row(**cell):
    return dict(cell)


# ─────────────────────────────────────────────────────────────────────────────
# min_n gate
# ─────────────────────────────────────────────────────────────────────────────

def test_run_family_insufficient_n_is_reported_not_silently_null() -> None:
    comps = [Comparison(name="c1", arm_a="a", arm_b="b", metric="f1")]
    data = {
        "a": [_row(f1=0.5), _row(f1=0.6)],
        "b": [_row(f1=0.3), _row(f1=0.4), _row(f1=0.5), _row(f1=0.6),
               _row(f1=0.7), _row(f1=0.8)],
    }
    report = run_family(family="H1b", comparisons=comps,
                         data_by_arm=data, min_n=5, alpha=0.05)
    r = report.results[0]
    assert r.status == "insufficient_n"
    assert r.p_value is None
    assert r.survives_holm is None
    assert report.n_tested == 0


def test_run_family_empty_arm_reports_empty_arm() -> None:
    comps = [Comparison(name="c1", arm_a="a", arm_b="b", metric="f1")]
    data = {"a": [], "b": [_row(f1=0.4)]}
    report = run_family(family="X", comparisons=comps, data_by_arm=data)
    assert report.results[0].status == "empty_arm"


# ─────────────────────────────────────────────────────────────────────────────
# Holm bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

def test_holm_survives_flagged_correctly_on_clear_effect() -> None:
    # Non-overlapping arms → MW U gives very small p.
    comps = [Comparison(name="c1", arm_a="a", arm_b="b", metric="f1")]
    data = {
        "a": [_row(f1=0.1), _row(f1=0.1), _row(f1=0.1),
               _row(f1=0.1), _row(f1=0.1)],
        "b": [_row(f1=0.9), _row(f1=0.9), _row(f1=0.9),
               _row(f1=0.9), _row(f1=0.9)],
    }
    report = run_family(family="X", comparisons=comps, data_by_arm=data,
                         min_n=5, alpha=0.05)
    r = report.results[0]
    assert r.status == "ok"
    assert r.p_value is not None and r.p_value < 0.05
    assert r.survives_holm is True
    assert r.borderline_nonsurvivor is False
    assert r.cliffs_delta is not None and abs(r.cliffs_delta) > 0.9
    assert r.delta_magnitude == "large"


def test_holm_borderline_reported_when_family_absorbs_significance() -> None:
    """One nominally significant comparison alongside many null tests may fail Holm."""
    comps: list[Comparison] = []
    data: dict[str, list[dict]] = {}
    # 1 real difference + 9 identical-arm null pairs to inflate the family size.
    comps.append(Comparison(name="real", arm_a="real_a", arm_b="real_b", metric="f1"))
    data["real_a"] = [_row(f1=0.55), _row(f1=0.60), _row(f1=0.65),
                       _row(f1=0.66), _row(f1=0.67)]
    data["real_b"] = [_row(f1=0.45), _row(f1=0.50), _row(f1=0.51),
                       _row(f1=0.52), _row(f1=0.54)]
    for i in range(9):
        an, bn = f"n{i}_a", f"n{i}_b"
        comps.append(Comparison(name=f"null_{i}", arm_a=an, arm_b=bn, metric="f1"))
        # Identical distributions on both sides → MW U will report p ≈ 1.
        data[an] = [_row(f1=v) for v in (0.5, 0.5, 0.5, 0.5, 0.5)]
        data[bn] = [_row(f1=v) for v in (0.5, 0.5, 0.5, 0.5, 0.5)]

    report = run_family(family="X", comparisons=comps, data_by_arm=data,
                         min_n=5, alpha=0.05)
    real = next(r for r in report.results if r.name == "real")
    # p is small enough on its own …
    assert real.p_value is not None and real.p_value < 0.05
    # … but Holm across 10 hypotheses raises the bar; whether survives_holm
    # is True depends on the exact p — assert the accounting is consistent.
    if real.survives_holm is False:
        assert real.borderline_nonsurvivor is True
        assert report.n_borderline >= 1
        assert report.n_significant_pre >= 1
        assert report.n_significant_post == 0
    else:
        assert real.borderline_nonsurvivor is False


def test_cliffs_delta_and_mean_populated_on_ok_result() -> None:
    comps = [Comparison(name="c1", arm_a="a", arm_b="b", metric="f1")]
    data = {
        "a": [_row(f1=0.2), _row(f1=0.4), _row(f1=0.6), _row(f1=0.8), _row(f1=1.0)],
        "b": [_row(f1=0.1), _row(f1=0.2), _row(f1=0.3), _row(f1=0.4), _row(f1=0.5)],
    }
    report = run_family(family="X", comparisons=comps, data_by_arm=data,
                         min_n=5, alpha=0.05)
    r = report.results[0]
    assert r.mean_a == pytest.approx(0.6)
    assert r.mean_b == pytest.approx(0.3)
    assert r.cliffs_delta is not None
    assert r.cliffs_delta > 0    # arm a stochastically dominates arm b
