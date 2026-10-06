"""declared comparison families with Holm correction.

The evaluation spec (§6, D04) requires:

* every planned comparison declared **before** scoring;
* Mann–Whitney U as the primary test;
* Cliff's δ as the effect-size companion;
* Holm correction over the *complete declared family*, not per-hypothesis;
* an explicit **minimum-n gate** — comparisons with too few observations on
  either side are ``not_applicable``, not silently reported as null;
* a list of **borderline non-survivors** — comparisons that were nominally
  significant but did not survive Holm correction, reported as evidence
  they did not survive rather than dropped.

This module wraps :mod:`fame.evaluation.stats` (which owns the numeric
primitives) and produces reproducible per-family tables. It never mutates
input CSVs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from fame.evaluation.stats import (
    cliffs_delta, cliffs_delta_magnitude, holm_correct, mann_whitney_u,
)


@dataclass(frozen=True)
class Comparison:
    """Declaration of one planned comparison inside a family."""
    name: str                       # human label, e.g. "H1b_repair_N10_rag_vs_nonrag"
    arm_a: str                      # name of the "left" arm
    arm_b: str                      # name of the "right" arm
    metric: str                     # wide-column name to compare


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    metric: str
    n_a: int
    n_b: int
    mean_a: Optional[float]
    mean_b: Optional[float]
    p_value: Optional[float]
    p_adjusted: Optional[float]
    cliffs_delta: Optional[float]
    delta_magnitude: Optional[str]
    survives_holm: Optional[bool]
    borderline_nonsurvivor: bool
    status: str                     # "ok", "insufficient_n", "empty_arm"
    reason: str = ""


@dataclass(frozen=True)
class FamilyReport:
    family: str
    alpha: float
    min_n: int
    results: List[ComparisonResult]
    n_tested: int                   # comparisons that reached Holm
    n_significant_pre: int          # nominally significant at raw p<alpha
    n_significant_post: int         # significant after Holm
    n_borderline: int               # nominally significant but did not survive Holm


def _finite(values: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return out


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def run_family(
    *,
    family: str,
    comparisons: Sequence[Comparison],
    data_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    min_n: int = 5,
    alpha: float = 0.05,
) -> FamilyReport:
    """Test eligible comparisons; preserve all declared slots in Holm.

    ``data_by_arm`` maps an arm name (matching :class:`Comparison`'s
    ``arm_a`` / ``arm_b``) to its list of run rows (usually the campaign aggregator's
    wide.csv rows filtered to that arm's cell). Each row is a dict; the
    metric value is looked up by key.
    """
    preliminary: List[ComparisonResult] = []
    testable_p: List[float] = []
    testable_indices: List[int] = []

    for idx, comp in enumerate(comparisons):
        rows_a = data_by_arm.get(comp.arm_a, [])
        rows_b = data_by_arm.get(comp.arm_b, [])
        values_a = _finite(row.get(comp.metric) for row in rows_a)
        values_b = _finite(row.get(comp.metric) for row in rows_b)
        n_a, n_b = len(values_a), len(values_b)

        if n_a == 0 or n_b == 0:
            preliminary.append(ComparisonResult(
                name=comp.name, metric=comp.metric, n_a=n_a, n_b=n_b,
                mean_a=_mean_or_none(values_a), mean_b=_mean_or_none(values_b),
                p_value=None, p_adjusted=None,
                cliffs_delta=None, delta_magnitude=None,
                survives_holm=None, borderline_nonsurvivor=False,
                status="empty_arm",
                reason=f"arm {comp.arm_a!r}: n={n_a}, arm {comp.arm_b!r}: n={n_b}",
            ))
            continue

        if n_a < min_n or n_b < min_n:
            preliminary.append(ComparisonResult(
                name=comp.name, metric=comp.metric, n_a=n_a, n_b=n_b,
                mean_a=_mean_or_none(values_a), mean_b=_mean_or_none(values_b),
                p_value=None, p_adjusted=None,
                cliffs_delta=None, delta_magnitude=None,
                survives_holm=None, borderline_nonsurvivor=False,
                status="insufficient_n",
                reason=f"min_n={min_n}; observed n_a={n_a}, n_b={n_b}",
            ))
            continue

        mw = mann_whitney_u(values_a, values_b)
        delta = cliffs_delta(values_a, values_b)
        preliminary.append(ComparisonResult(
            name=comp.name, metric=comp.metric, n_a=n_a, n_b=n_b,
            mean_a=_mean_or_none(values_a), mean_b=_mean_or_none(values_b),
            p_value=mw.p_value, p_adjusted=None,
            cliffs_delta=delta,
            delta_magnitude=cliffs_delta_magnitude(delta),
            survives_holm=None, borderline_nonsurvivor=False,
            status="ok",
        ))
        testable_p.append(mw.p_value)
        testable_indices.append(len(preliminary) - 1)

    if testable_p:
        # Retain the declared multiplicity even when a comparison is untestable.
        # Its internal placeholder p=1 is never published as an observed p-value.
        all_p = [r.p_value if r.p_value is not None else 1.0 for r in preliminary]
        adjusted = holm_correct(all_p, alpha=alpha)
        for slot_idx in testable_indices:
            adj = adjusted[slot_idx]
            prior = preliminary[slot_idx]
            raw_p = prior.p_value
            preliminary[slot_idx] = ComparisonResult(
                **{**prior.__dict__,
                   "p_adjusted": adj.adj_p,
                   "survives_holm": adj.reject,
                   "borderline_nonsurvivor": (raw_p is not None and raw_p < alpha and not adj.reject)},
            )

    n_significant_pre = sum(
        1 for r in preliminary if r.status == "ok" and r.p_value is not None and r.p_value < alpha
    )
    n_significant_post = sum(
        1 for r in preliminary if r.survives_holm is True
    )
    n_borderline = sum(1 for r in preliminary if r.borderline_nonsurvivor)

    return FamilyReport(
        family=family,
        alpha=alpha,
        min_n=min_n,
        results=preliminary,
        n_tested=len(testable_p),
        n_significant_pre=n_significant_pre,
        n_significant_post=n_significant_post,
        n_borderline=n_borderline,
    )
