"""Unit tests for fame/evaluation/stats.py."""
from __future__ import annotations

import math

import pytest

from fame.evaluation.stats import (
    Comparison,
    HolmResult,
    MannWhitneyResult,
    Summary,
    cliffs_delta,
    cliffs_delta_magnitude,
    compare,
    holm_correct,
    mann_whitney_u,
    summarise,
)


# ─────────────────────────────────────────────────────────────────────────────
# summarise
# ─────────────────────────────────────────────────────────────────────────────

def test_summarise_single_value_has_zero_sd() -> None:
    s = summarise([42.0])
    assert s.n == 1
    assert s.mean == 42.0
    assert s.sd == 0.0
    assert s.ci_low == s.ci_high == 42.0


def test_summarise_mean_and_sd() -> None:
    s = summarise([2, 4, 4, 4, 5, 5, 7, 9])
    assert s.n == 8
    assert s.mean == pytest.approx(5.0)
    # Sample SD
    assert s.sd == pytest.approx(2.138, abs=1e-3)


def test_summarise_ci_bounds_the_mean() -> None:
    s = summarise([10] * 10)
    assert s.ci_low == 10.0 and s.ci_high == 10.0     # zero variance = point CI


def test_summarise_median_odd_and_even() -> None:
    assert summarise([1, 2, 3]).median == 2.0
    assert summarise([1, 2, 3, 4]).median == 2.5


def test_summarise_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarise([])


# ─────────────────────────────────────────────────────────────────────────────
# Mann–Whitney U
# ─────────────────────────────────────────────────────────────────────────────

def test_mw_u_identical_samples() -> None:
    """Two identical samples → maximum p (≈ 1)."""
    r = mann_whitney_u([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert r.p_value == pytest.approx(1.0, abs=0.05)


def test_mw_u_clearly_separated_gives_small_p() -> None:
    a = [1, 2, 3, 4, 5, 6]
    b = [10, 11, 12, 13, 14, 15]
    r = mann_whitney_u(a, b)
    assert r.p_value < 0.05


def test_mw_u_symmetric() -> None:
    """Two-sided test — swapping a/b must not change p."""
    a = [1, 2, 3, 4, 5]
    b = [3, 4, 5, 6, 7]
    r1 = mann_whitney_u(a, b)
    r2 = mann_whitney_u(b, a)
    assert r1.p_value == pytest.approx(r2.p_value, abs=1e-9)


def test_mw_u_rejects_empty_side() -> None:
    with pytest.raises(ValueError):
        mann_whitney_u([], [1, 2, 3])
    with pytest.raises(ValueError):
        mann_whitney_u([1, 2, 3], [])


# ─────────────────────────────────────────────────────────────────────────────
# Holm correction
# ─────────────────────────────────────────────────────────────────────────────

def test_holm_single_p_unchanged() -> None:
    out = holm_correct([0.02])
    assert len(out) == 1
    assert out[0].adj_p == pytest.approx(0.02)
    assert out[0].reject is True   # 0.02 <= 0.05 / 1


def test_holm_all_significant() -> None:
    """All four raw p tiny → all rejected."""
    out = holm_correct([0.001, 0.002, 0.003, 0.004], alpha=0.05)
    assert all(r.reject for r in out)


def test_holm_step_down_stops_at_first_failure() -> None:
    """The classic Holm example: sorted p's (0.01, 0.04, 0.05, 0.20) vs alpha=0.05.

    Divisors: 4, 3, 2, 1 → thresholds: 0.0125, 0.0167, 0.025, 0.05.
    0.01 ≤ 0.0125 → reject
    0.04 > 0.0167 → stop; everything else NOT rejected."""
    out = holm_correct([0.01, 0.04, 0.05, 0.20])
    rejects = {r.raw_p: r.reject for r in out}
    assert rejects[0.01] is True
    assert rejects[0.04] is False
    assert rejects[0.05] is False
    assert rejects[0.20] is False


def test_holm_returns_in_input_order() -> None:
    out = holm_correct([0.5, 0.001, 0.3, 0.002])
    # Verify index preserved
    assert [r.index for r in out] == [0, 1, 2, 3]
    assert [r.raw_p for r in out] == [0.5, 0.001, 0.3, 0.002]


def test_holm_adj_p_monotone_after_sorting() -> None:
    out = holm_correct([0.5, 0.001, 0.3, 0.002])
    sorted_by_raw = sorted(out, key=lambda r: r.raw_p)
    adj = [r.adj_p for r in sorted_by_raw]
    assert all(a <= b for a, b in zip(adj, adj[1:])), \
        f"adj p not monotone: {adj}"


def test_holm_rejects_out_of_range_p() -> None:
    with pytest.raises(ValueError):
        holm_correct([0.5, 1.5])
    with pytest.raises(ValueError):
        holm_correct([-0.1, 0.5])


def test_holm_labels_are_carried() -> None:
    out = holm_correct([0.001, 0.5], labels=["primary", "secondary"])
    assert out[0].label == "primary"
    assert out[1].label == "secondary"


def test_holm_label_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        holm_correct([0.1, 0.2], labels=["a"])


def test_holm_empty_input_ok() -> None:
    assert holm_correct([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Cliff's delta
# ─────────────────────────────────────────────────────────────────────────────

def test_cliffs_delta_all_a_greater_gives_plus_one() -> None:
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0


def test_cliffs_delta_all_a_less_gives_minus_one() -> None:
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0


def test_cliffs_delta_identical_gives_zero() -> None:
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_cliffs_delta_ties_neutral() -> None:
    """Ties count for neither side.

    a=[1,2,3] vs b=[2,3,4] → gt=1 (only 3>2), lt=6, ties=2. δ = (1-6)/9 = -5/9.
    """
    assert cliffs_delta([1, 2, 3], [2, 3, 4]) == pytest.approx(-5 / 9, abs=1e-6)


def test_cliffs_delta_magnitude_labels() -> None:
    assert cliffs_delta_magnitude(0.10) == "negligible"
    assert cliffs_delta_magnitude(0.20) == "small"
    assert cliffs_delta_magnitude(0.40) == "medium"
    assert cliffs_delta_magnitude(0.60) == "large"
    assert cliffs_delta_magnitude(-0.60) == "large"


def test_cliffs_delta_rejects_empty() -> None:
    with pytest.raises(ValueError):
        cliffs_delta([], [1, 2])


# ─────────────────────────────────────────────────────────────────────────────
# compare() bundle
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_bundles_summary_mw_cliffs() -> None:
    r = compare([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], label="A vs B")
    assert isinstance(r.a_summary, Summary)
    assert isinstance(r.b_summary, Summary)
    assert isinstance(r.mw, MannWhitneyResult)
    assert r.cliffs == -1.0
    assert r.cliffs_label == "large"
    assert r.mw.p_value < 0.05
