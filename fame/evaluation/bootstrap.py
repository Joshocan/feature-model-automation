"""deterministic percentile bootstrap confidence intervals.

The evaluation spec (§6) requires means with 95% intervals alongside every
headline number. This module provides the primitive: percentile bootstrap
with a pinned seed so the same input always produces the same CI. No
scipy/numpy random-state dependencies — the reproducibility rules for the
paper table forbid uncontrolled RNG drift.

Envelope semantics
------------------
The helpers accept iterables of raw floats. Callers must strip the
``status="ok"`` gate upstream (the campaign aggregator's ``_numeric_ok_values`` does this).
This module does not silently swallow ``None``s — an unknown value is a
denominator problem for the caller, not a bootstrap concern.
"""
from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BootstrapCI:
    """A percentile bootstrap CI for one summary statistic."""
    statistic: str
    point: float
    lower: float
    upper: float
    confidence: float
    n: int
    n_resamples: int
    seed: int


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = st.fmean,
    statistic_name: str = "mean",
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 20260927,
) -> Optional[BootstrapCI]:
    """Return a percentile-bootstrap CI or ``None`` if the input is too small.

    ``None`` is the honest answer for n<2; the paper table treats it as
    "interval not computable at this n" rather than reporting a zero-width
    band.
    """
    n = len(values)
    if n < 2:
        return None
    if any(v is None or (isinstance(v, float) and not math.isfinite(v)) for v in values):
        raise ValueError("bootstrap_ci received non-finite value; caller must filter")
    rng = random.Random(seed)
    resamples: List[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resamples.append(statistic(sample))
    resamples.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int((alpha / 2) * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return BootstrapCI(
        statistic=statistic_name,
        point=float(statistic(values)),
        lower=float(resamples[lo_idx]),
        upper=float(resamples[hi_idx]),
        confidence=confidence,
        n=n,
        n_resamples=n_resamples,
        seed=seed,
    )


def mean_ci(values: Sequence[float], **kwargs) -> Optional[BootstrapCI]:
    """Convenience wrapper — the mean is what §6 explicitly requires."""
    return bootstrap_ci(values, statistic=st.fmean, statistic_name="mean", **kwargs)


def median_ci(values: Sequence[float], **kwargs) -> Optional[BootstrapCI]:
    """Convenience wrapper for median CIs used in dispersion tables."""
    return bootstrap_ci(values, statistic=st.median, statistic_name="median", **kwargs)


def paired_difference_ci(
    a: Sequence[float],
    b: Sequence[float],
    **kwargs,
) -> Optional[BootstrapCI]:
    """CI on the paired difference ``mean(a) − mean(b)``.

    Pairing must be justified by design (§6). This helper does not check
    that; it just refuses inputs of different length.
    """
    if len(a) != len(b):
        raise ValueError("paired_difference_ci needs equal-length inputs")
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_ci(diffs, statistic=st.fmean,
                        statistic_name="mean_paired_difference", **kwargs)
