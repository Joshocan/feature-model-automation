from __future__ import annotations

import math

import pytest

from fame.evaluation.bootstrap import (
    bootstrap_ci, mean_ci, median_ci, paired_difference_ci,
)


def test_mean_ci_reproducible_with_pinned_seed() -> None:
    values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    a = mean_ci(values, n_resamples=2000, seed=42)
    b = mean_ci(values, n_resamples=2000, seed=42)
    assert a is not None and b is not None
    assert a.point == b.point
    assert a.lower == b.lower
    assert a.upper == b.upper


def test_mean_ci_different_seed_gives_different_bounds() -> None:
    values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    a = mean_ci(values, n_resamples=2000, seed=42)
    b = mean_ci(values, n_resamples=2000, seed=1234567)
    assert a is not None and b is not None
    # The bounds must not be identical across independent seeds.
    assert (a.lower, a.upper) != (b.lower, b.upper)


def test_mean_ci_returns_none_when_n_below_2() -> None:
    assert mean_ci([0.5]) is None
    assert mean_ci([]) is None


def test_bootstrap_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_ci([1.0, 2.0, float("nan")])


def test_median_ci_covers_true_median_on_wide_distribution() -> None:
    values = [i / 100 for i in range(100)]  # 0.00 … 0.99
    ci = median_ci(values, n_resamples=2000, seed=17)
    assert ci is not None
    assert ci.lower <= 0.495 <= ci.upper


def test_paired_difference_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_difference_ci([0.1, 0.2, 0.3], [0.4, 0.5])


def test_paired_difference_ci_of_shifted_pairs() -> None:
    # Diffs of 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 — genuinely different values so
    # the bootstrap distribution has spread and the interval has non-zero width.
    a = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    b = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    diffs = [x - y for x, y in zip(a, b)]
    ci = paired_difference_ci(a, b, n_resamples=2000, seed=7)
    assert ci is not None
    expected_mean = sum(diffs) / len(diffs)
    assert ci.point == pytest.approx(expected_mean)
    # Interval brackets the true (sample-mean) shift.
    assert ci.lower <= expected_mean <= ci.upper
    # And has non-zero width.
    assert ci.upper > ci.lower


def test_bootstrap_ci_point_matches_statistic_on_input() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = bootstrap_ci(values, statistic=sum, statistic_name="sum",
                       n_resamples=1000, seed=99)
    assert ci is not None
    assert ci.point == 15.0
    assert ci.n == 5


def test_bootstrap_confidence_level_widens_interval() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tight = mean_ci(values, confidence=0.80, n_resamples=2000, seed=3)
    loose = mean_ci(values, confidence=0.99, n_resamples=2000, seed=3)
    assert tight is not None and loose is not None
    width_tight = tight.upper - tight.lower
    width_loose = loose.upper - loose.lower
    assert width_loose > width_tight
