"""τ re-thresholding of the raw cosine-similarity pairs table.

The semantic evaluator wrote every generated↔reference cosine similarity to ``pairs.csv``
without applying any τ cutoff. This module re-scores those raw similarities
at any τ (and optionally under either the independent-max or one-to-one
matching policy) so the paper's τ-sweep table falls out of one pass over
that file.

The pipeline is intentionally simple: read pairs, group by run, threshold,
apply the matching policy, compute P / R_total / F1_total. Downstream code
composes dual recall (against reach / attested / organising) by joining
this output with the semantic evaluator's partition metadata.

Nothing here recomputes cosine; nothing embedding-related is imported.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TauScore:
    """Per-run scoring at one τ under one matching policy."""
    run_id: str
    tau: float
    matching_policy: str
    n_generated: int
    n_reference: int
    n_generated_matched: int
    n_reference_matched: int
    n_matched_pairs: int
    precision: float
    recall_total: float
    f1_total: float
    matched_reference_ids: List[str]


def _group_pairs_by_run(rows: Iterable[Mapping[str, Any]]
                        ) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = row.get("run_id")
        if rid is None:
            continue
        out[rid].append(row)
    return dict(out)


def _independent_max(pairs: Sequence[Mapping[str, Any]], tau: float
                      ) -> Tuple[int, int, int, List[str]]:
    """Independent max — the registered primary matching policy.

    Returns ``(n_generated_matched, n_reference_matched, n_matched_pairs,
    matched_reference_ids)``.
    """
    ref_by_id: Dict[int, float] = {}
    gen_by_id: Dict[int, float] = {}
    ref_name_by_id: Dict[int, str] = {}
    for pair in pairs:
        sim = float(pair.get("similarity", 0.0) or 0.0)
        if sim < tau:
            continue
        ref = int(pair["ref_index"])
        gen = int(pair["gen_index"])
        ref_by_id[ref] = max(ref_by_id.get(ref, 0.0), sim)
        gen_by_id[gen] = max(gen_by_id.get(gen, 0.0), sim)
        ref_name_by_id[ref] = str(pair.get("ref_name", ""))
    return (len(gen_by_id), len(ref_by_id), -1,
            [ref_name_by_id[i] for i in sorted(ref_by_id)])


def _one_to_one(pairs: Sequence[Mapping[str, Any]], tau: float, *, cardinality_first: bool = True
                 ) -> Tuple[int, int, int, List[str]]:
    """Thresholded bipartite assignment with an explicit matching objective.

    Cardinality first is the existing sensitivity policy, now with maximum
    weight as its secondary objective. Weight-only is separately labelled.
    Returns the same
    tuple shape as :func:`_independent_max` with
    ``n_generated_matched == n_reference_matched == n_matched_pairs``.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    gen_ids = sorted({int(p["gen_index"]) for p in pairs})
    ref_ids = sorted({int(p["ref_index"]) for p in pairs})
    gen_index = {g: i for i, g in enumerate(gen_ids)}
    ref_index = {r: j for j, r in enumerate(ref_ids)}
    ref_name_by_id = {int(p["ref_index"]): str(p["ref_name"]) for p in pairs}
    matrix = np.full((len(gen_ids), len(ref_ids)), -np.inf, dtype=float)
    for pair in pairs:
        matrix[gen_index[int(pair["gen_index"])], ref_index[int(pair["ref_index"])]] = \
            float(pair.get("similarity", 0.0) or 0.0)
    # Dummy columns permit unmatched nodes. Bonus makes cardinality primary;
    # cosine weight then breaks ties without changing P/R/F1 under that policy.
    bonus = 2 * min(matrix.shape) + 1 if cardinality_first else 0
    weights = np.zeros((len(gen_ids), len(ref_ids) + len(gen_ids)))
    weights[:, :len(ref_ids)] = np.where(matrix >= tau, bonus + matrix, -1e12)
    gi, ri = linear_sum_assignment(weights, maximize=True)
    matched = [(int(i), int(j)) for i, j in zip(gi, ri)
               if j < len(ref_ids) and matrix[i, j] >= tau]
    matched_refs = [ref_name_by_id[ref_ids[j]] for _, j in sorted(matched)]
    n = len(matched)
    return (n, n, n, matched_refs)


def rescore_run(
    pairs: Sequence[Mapping[str, Any]],
    *,
    tau: float,
    matching_policy: str = "independent_max",
) -> TauScore:
    """Score one run's pairs at ``tau`` under ``matching_policy``.

    Empty pair lists → all counts zero and P/R/F1 = 0 (a measured zero,
    not a status marker). Distinguishing "no features" from "no matches"
    is a job for the semantic evaluator's envelope, not this pass.
    """
    gen_ids = {int(p["gen_index"]) for p in pairs}
    ref_ids = {int(p["ref_index"]) for p in pairs}
    n_gen = len(gen_ids)
    n_ref = len(ref_ids)

    if not pairs or n_gen == 0 or n_ref == 0:
        return TauScore(
            run_id=str(pairs[0].get("run_id", "")) if pairs else "",
            tau=tau, matching_policy=matching_policy,
            n_generated=n_gen, n_reference=n_ref,
            n_generated_matched=0, n_reference_matched=0, n_matched_pairs=0,
            precision=0.0, recall_total=0.0, f1_total=0.0,
            matched_reference_ids=[],
        )

    if matching_policy == "independent_max":
        n_gen_hit, n_ref_hit, n_paired, matched_refs = _independent_max(pairs, tau)
    elif matching_policy == "one_to_one":
        n_gen_hit, n_ref_hit, n_paired, matched_refs = _one_to_one(pairs, tau)
    elif matching_policy == "one_to_one_max_weight":
        n_gen_hit, n_ref_hit, n_paired, matched_refs = _one_to_one(pairs, tau, cardinality_first=False)
    else:
        raise ValueError(f"unknown matching_policy: {matching_policy!r}")

    precision = n_gen_hit / n_gen if n_gen else 0.0
    recall = n_ref_hit / n_ref if n_ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return TauScore(
        run_id=str(pairs[0].get("run_id", "")),
        tau=tau, matching_policy=matching_policy,
        n_generated=n_gen, n_reference=n_ref,
        n_generated_matched=n_gen_hit, n_reference_matched=n_ref_hit,
        n_matched_pairs=n_paired,
        precision=precision, recall_total=recall, f1_total=f1,
        matched_reference_ids=matched_refs,
    )


def rescore_all(
    pair_rows: Iterable[Mapping[str, Any]],
    *,
    taus: Sequence[float] = (0.3, 0.4, 0.5, 0.6),
    matching_policies: Sequence[str] = ("independent_max",),
) -> List[TauScore]:
    """Cartesian product of runs × τ × policies."""
    by_run = _group_pairs_by_run(pair_rows)
    out: List[TauScore] = []
    for run_id in sorted(by_run):
        for tau in taus:
            for policy in matching_policies:
                score = rescore_run(by_run[run_id], tau=tau,
                                    matching_policy=policy)
                out.append(score)
    return out
