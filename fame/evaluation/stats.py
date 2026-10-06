"""Statistical hygiene helpers for cross-run analysis.

Nothing here calls an LLM. Everything is a pure function over numeric arrays,
suitable for unit testing with synthetic data. The functions implement the
statistical practice the IFS brief mandates in §10.10:

* Mann–Whitney U (unpaired, two-sided) for comparing configurations
* Holm correction across the whole declared family of tests
* Cliff's delta as the effect size reported alongside adjusted p
* Mean with 95% confidence interval + SD for per-configuration dispersion

Design goals
------------
* Zero silent failures — every helper raises ``ValueError`` on unusable input
  rather than returning a NaN.
* Determinism — no random state anywhere.
* No hidden knobs — every threshold / method / correction is a keyword arg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Summary:
    """Central tendency + dispersion + 95% CI over one configuration's values."""
    n: int
    mean: float
    sd: float
    ci_low: float
    ci_high: float
    median: float
    min: float
    max: float


@dataclass(frozen=True)
class MannWhitneyResult:
    """Unpaired two-sided Mann–Whitney U with per-side sizes."""
    u: float                # U statistic (the smaller of U_a, U_b)
    p_value: float          # two-sided
    n_a: int
    n_b: int
    method: str = "mann-whitney-u-two-sided"


@dataclass(frozen=True)
class HolmResult:
    """One entry in a Holm-corrected family."""
    index:   int
    raw_p:   float
    adj_p:   float
    reject:  bool           # reject H0 at the family-wise alpha
    label:   Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Central tendency + dispersion
# ─────────────────────────────────────────────────────────────────────────────

def summarise(values: Sequence[float], *, confidence: float = 0.95) -> Summary:
    """Descriptive statistics + normal-approx CI on the mean.

    We deliberately use the normal approximation (z = 1.96 for 95%) rather
    than a t-interval because repetition counts in the campaign (5, 20, ...)
    are always ≥ 5, and the campaign reports the SD alongside so anyone who
    needs the exact t-interval can recompute.
    """
    if not values:
        raise ValueError("summarise() requires at least one value")
    n = len(values)
    xs = [float(x) for x in values]
    xs_sorted = sorted(xs)
    mean = sum(xs) / n
    if n == 1:
        sd = 0.0
    else:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        sd = math.sqrt(var)
    z = _z_for(confidence)
    half = z * (sd / math.sqrt(n)) if n > 1 else 0.0
    return Summary(
        n=n,
        mean=mean,
        sd=sd,
        ci_low=mean - half,
        ci_high=mean + half,
        median=xs_sorted[n // 2] if n % 2 else 0.5 * (xs_sorted[n // 2 - 1] + xs_sorted[n // 2]),
        min=xs_sorted[0],
        max=xs_sorted[-1],
    )


def _z_for(confidence: float) -> float:
    """Two-sided normal critical value.

    Table lookup keeps this pure-Python (no scipy import at module load).
    """
    lookup = {
        0.90: 1.6449,
        0.95: 1.9600,
        0.99: 2.5758,
    }
    if confidence in lookup:
        return lookup[confidence]
    raise ValueError(f"unsupported confidence level: {confidence!r}. "
                     f"Supported: {sorted(lookup)}")


# ─────────────────────────────────────────────────────────────────────────────
# Mann–Whitney U — unpaired, two-sided, with ties
# ─────────────────────────────────────────────────────────────────────────────

def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> MannWhitneyResult:
    """Two-sided Mann–Whitney U test.

    Uses SciPy when available (exact for small n, normal approx otherwise);
    falls back to a pure-Python implementation using tie-corrected rank sums.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        raise ValueError("mann_whitney_u requires non-empty samples on both sides")
    pooled = [float(value) for value in (*a, *b)]
    if any(not math.isfinite(value) for value in pooled):
        raise ValueError("mann_whitney_u requires finite samples")
    # SciPy versions can return NaN for the zero-variance, all-tied case.
    # There is no between-group rank difference, so the two-sided p is 1.
    if min(pooled) == max(pooled):
        return MannWhitneyResult(u=n_a * n_b / 2, p_value=1.0,
                                 n_a=n_a, n_b=n_b)

    # Preferred path — SciPy handles ties + exact/asymptotic seamlessly.
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
        stat = mannwhitneyu(list(a), list(b), alternative="two-sided")
        u = min(float(stat.statistic), float(n_a * n_b - stat.statistic))
        p = float(stat.pvalue)
        if math.isfinite(p):
            return MannWhitneyResult(u=u, p_value=p, n_a=n_a, n_b=n_b)
    except ImportError:
        pass  # fall through to the manual implementation

    return _mann_whitney_manual(a, b)


def _mann_whitney_manual(a: Sequence[float], b: Sequence[float]) -> MannWhitneyResult:
    """Pure-Python fallback with tie correction + normal approximation."""
    n_a, n_b = len(a), len(b)
    combined = [(float(v), 0) for v in a] + [(float(v), 1) for v in b]
    combined.sort(key=lambda t: t[0])

    # Assign ranks with tie averaging
    ranks: List[float] = [0.0] * len(combined)
    i = 0
    tie_correction = 0.0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_correction += t ** 3 - t
        i = j + 1

    r_a = sum(rank for rank, (_, side) in zip(ranks, combined) if side == 0)
    u_a = r_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    mu = n_a * n_b / 2
    n = n_a + n_b
    sigma_sq = (n_a * n_b / 12) * ((n + 1) - tie_correction / (n * (n - 1)))
    if sigma_sq <= 0:
        return MannWhitneyResult(u=u, p_value=1.0, n_a=n_a, n_b=n_b)
    z = (u - mu) / math.sqrt(sigma_sq)
    # Two-sided p from normal CDF
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return MannWhitneyResult(u=u, p_value=max(0.0, min(1.0, p)), n_a=n_a, n_b=n_b)


def _normal_cdf(x: float) -> float:
    """Φ(x) via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Holm correction — one for the whole declared family
# ─────────────────────────────────────────────────────────────────────────────

def holm_correct(p_values: Sequence[float],
                 *,
                 labels: Optional[Sequence[str]] = None,
                 alpha: float = 0.05) -> List[HolmResult]:
    """Holm–Bonferroni step-down correction.

    Given ``m`` raw p-values, sort ascending, compare the k-th smallest
    against ``alpha / (m - k)``. First failure stops the chain — all
    subsequent p-values are also not rejected.

    Returns entries in the original input order (not sorted).
    """
    if labels is not None and len(labels) != len(p_values):
        raise ValueError(f"labels length {len(labels)} != p_values length {len(p_values)}")
    m = len(p_values)
    if m == 0:
        return []
    if any(not (0.0 <= p <= 1.0) for p in p_values):
        raise ValueError("all p_values must be in [0, 1]")

    # Sort indices by ascending p
    order = sorted(range(m), key=lambda i: p_values[i])
    reject_flags = [False] * m
    adj_ps: List[float] = [0.0] * m
    running_max = 0.0
    stopped = False
    for rank, idx in enumerate(order):
        divisor = m - rank
        adj = min(1.0, p_values[idx] * divisor)
        # Enforce monotonicity: adjusted p must be non-decreasing along sorted order
        running_max = max(running_max, adj)
        adj_ps[idx] = running_max
        if not stopped and p_values[idx] <= alpha / divisor:
            reject_flags[idx] = True
        else:
            stopped = True

    out: List[HolmResult] = []
    for i in range(m):
        out.append(HolmResult(
            index=i,
            raw_p=float(p_values[i]),
            adj_p=float(adj_ps[i]),
            reject=reject_flags[i],
            label=(labels[i] if labels else None),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cliff's delta — effect size for two independent samples
# ─────────────────────────────────────────────────────────────────────────────

def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's δ = P(A > B) - P(A < B).

    Range: -1 (all A < all B) .. 0 (no separation) .. +1 (all A > all B).
    Ties count for neither side.
    """
    if not a or not b:
        raise ValueError("cliffs_delta requires non-empty samples on both sides")
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    n_a, n_b = len(a), len(b)
    total = n_a * n_b
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:  gt += 1
            elif x < y: lt += 1
    return (gt - lt) / total


def cliffs_delta_magnitude(delta: float) -> str:
    """Categorical label per Romano et al. thresholds: negligible / small / medium / large."""
    d = abs(delta)
    if d < 0.147:  return "negligible"
    if d < 0.33:   return "small"
    if d < 0.474:  return "medium"
    return "large"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience — combined MW U + Cliff's δ + summary for one comparison
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Comparison:
    """A single MW U comparison + effect size + per-side summaries."""
    label: str
    a_summary: Summary
    b_summary: Summary
    mw: MannWhitneyResult
    cliffs: float
    cliffs_label: str


def compare(a: Sequence[float], b: Sequence[float], *,
            label: str = "A vs B") -> Comparison:
    return Comparison(
        label=label,
        a_summary=summarise(a),
        b_summary=summarise(b),
        mw=mann_whitney_u(a, b),
        cliffs=cliffs_delta(a, b),
        cliffs_label=cliffs_delta_magnitude(cliffs_delta(a, b)),
    )
