"""criterion-divergence rank table.

Ranks each campaign cell separately under three criteria:

* **structural** — mean of ``structural__structural_conformance``
  (fraction of runs in cell whose XSD/W1–W5 tree gate passed).
* **logical**    — mean of ``structural__satisfiable`` (fraction SAT)
  and inverse of ``structural__dead_feature_ratio`` if desired; the
  default is the SAT rate alone.
* **semantic**   — mean of ``semantic__semantic_f1_total`` at τ_primary
  (independent-max).

Rank correlations (Spearman ρ, Kendall τ) between criterion rankings
answer the paper's central load-bearing question: **do the criteria
disagree, or do they collapse into one order?** If the rank correlations
are near 1, the separation claim is decorative.

The module is deliberately deterministic and pure — no plots, no
Matplotlib; it produces rank tables + correlation coefficients so
downstream analysis (or paper tables) can render them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CriterionRanking:
    """One criterion's ranking. Ties share the same rank (average method)."""
    criterion: str
    cell_ids: List[Tuple[Any, ...]]  # in descending-score order
    scores:   List[Optional[float]]  # None → cell had no ok observations
    ranks:    List[float]            # 1 = best; ties get averaged ranks


@dataclass(frozen=True)
class RankCorrelation:
    """Pairwise correlation between two rankings."""
    criterion_a: str
    criterion_b: str
    n: int
    spearman_rho: Optional[float]
    kendall_tau:  Optional[float]
    n_excluded: int = 0
    status: str = "ok"


def _valid_pairs(a, b):
    """Use the same finite, observed pairs for both correlation coefficients."""
    if len(a) != len(b):
        raise ValueError("Correlation inputs must have equal lengths")
    return [(x, y) for x, y in zip(a, b)
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]


def _average_ranks(scores: Sequence[Optional[float]]) -> List[float]:
    """Assign ranks 1..n with tie averaging; ``None`` scores go last."""
    indexed = [(i, s) for i, s in enumerate(scores)]

    def _key(item):
        i, s = item
        return (s is None, -(s if s is not None else 0.0), i)

    ordered = sorted(indexed, key=_key)
    n = len(ordered)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2 or len(a) != len(b):
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def spearman_rho(scores_a: Sequence[Optional[float]], scores_b: Sequence[Optional[float]]
                 ) -> Optional[float]:
    """Pearson correlation over average ranks — the classical Spearman ρ."""
    pairs = _valid_pairs(scores_a, scores_b)
    ranks_a = _average_ranks([a for a, _ in pairs])
    ranks_b = _average_ranks([b for _, b in pairs])
    return _pearson(ranks_a, ranks_b)


def kendall_tau(scores_a: Sequence[Optional[float]], scores_b: Sequence[Optional[float]]
                 ) -> Optional[float]:
    """Kendall τ-b with tie correction. Pairs where either side is None are dropped."""
    pairs = _valid_pairs(scores_a, scores_b)
    n = len(pairs)
    if n < 2:
        return None
    concordant = discordant = tied_a = tied_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = pairs[i][0] - pairs[j][0]
            db = pairs[i][1] - pairs[j][1]
            if da == 0 and db == 0:
                tied_a += 1
                tied_b += 1
                continue
            if da == 0:
                tied_a += 1
                continue
            if db == 0:
                tied_b += 1
                continue
            sign_a = da > 0
            sign_b = db > 0
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - tied_a) * (n0 - tied_b))
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def build_rankings(
    cell_rows: Sequence[Mapping[str, Any]],
    *,
    cell_id_keys: Sequence[str],
    criteria: Mapping[str, str],
) -> List[CriterionRanking]:
    """Build one :class:`CriterionRanking` per criterion.

    ``criteria`` maps a display name (``"structural"``, ``"logical"``,
    ``"semantic"``) to the *summary column* to sort by
    (e.g. ``"structural__structural_conformance__mean"``). Cells whose
    column value is ``None`` land at the tail of the ranking with a real
    (largest) rank, so downstream correlations can drop them.
    """
    cell_ids = [tuple(row.get(k) for k in cell_id_keys) for row in cell_rows]
    rankings: List[CriterionRanking] = []
    for name, col in criteria.items():
        scores = [row.get(col) for row in cell_rows]
        ranks = _average_ranks(scores)
        order = sorted(range(len(cell_rows)), key=lambda i: (ranks[i], cell_ids[i]))
        rankings.append(CriterionRanking(
            criterion=name,
            cell_ids=[cell_ids[i] for i in order],
            scores=[scores[i] for i in order],
            ranks=[ranks[i] for i in order],
        ))
    return rankings


def pairwise_correlations(
    cell_rows: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, str],
) -> List[RankCorrelation]:
    """Every unordered pair of criteria → Spearman ρ + Kendall τ."""
    names = list(criteria)
    scores_by_name = {name: [row.get(col) for row in cell_rows]
                       for name, col in criteria.items()}
    out: List[RankCorrelation] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs = _valid_pairs(scores_by_name[a], scores_by_name[b])
            n_pairs = len(pairs)
            status = ("insufficient_n" if n_pairs < 2 else
                      "constant_input" if len({x for x, _ in pairs}) < 2 or
                      len({y for _, y in pairs}) < 2 else "ok")
            out.append(RankCorrelation(
                criterion_a=a, criterion_b=b, n=n_pairs,
                n_excluded=len(cell_rows) - n_pairs, status=status,
                spearman_rho=spearman_rho(scores_by_name[a], scores_by_name[b]),
                kendall_tau=kendall_tau(scores_by_name[a], scores_by_name[b]),
            ))
    return out


DEFAULT_CRITERIA: Dict[str, str] = {
    "structural": "structural__structural_conformance__mean",
    "logical":    "structural__satisfiable__mean",
    "semantic":   "semantic__semantic_f1_total__mean",
}
