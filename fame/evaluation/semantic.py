from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from .coverage import extract_nodes


SEMANTIC_METRICS = (
    "n_reference", "n_generated",
    "semantic_precision", "semantic_recall_total", "semantic_f1_total",
    "recall_reach", "recall_attested", "recall_organising",
    "matched_reference_ids", "duplicate_generated_hits",
    "reach_size", "F_t_size", "F_t_attested_size", "F_t_organising_size",
    "rho_T_C",
    "n_parent_evaluable", "n_parent_correct", "parent_match_rate",
    "n_exact_parent_evaluable", "n_exact_parent_correct", "exact_name_parent_match_rate",
    "exact_duplicate_surplus", "near_duplicate_pairs_tau_0_8", "near_duplicate_pairs_tau_0_9",
    "n_citations", "n_attribution_applicable", "n_attribution_agreeing",
    "attribution_agreement_rate", "n_citations_unmapped", "n_citations_reference_unannotated",
)


def _envelope(value: Any = None, status: str = "ok", reason: str = "", **counts) -> dict:
    return dict(value=value, status=status, reason=reason, **counts)


def _ineligible(reason: str) -> dict:
    return {key: _envelope(status="ineligible", reason=reason) for key in SEMANTIC_METRICS}


def cosine_similarity_matrix(
    reference_names: Sequence[str],
    generated_names: Sequence[str],
    *,
    model,
) -> np.ndarray:
    """Return a generated x reference cosine-similarity matrix.

    The encoder must expose the sentence-transformers-compatible ``encode``
    method. Embeddings are requested normalised, so cosine similarity is the
    matrix product. Unlike :func:`semantic_prf`, this strict primitive does
    not hide encoder or XML errors.
    """
    if not reference_names or not generated_names:
        return np.empty((len(generated_names), len(reference_names)), dtype=float)
    reference = model.encode(
        list(reference_names), normalize_embeddings=True, convert_to_tensor=False
    )
    generated = model.encode(
        list(generated_names), normalize_embeddings=True, convert_to_tensor=False
    )
    return np.asarray(generated, dtype=float) @ np.asarray(reference, dtype=float).T


def maximum_threshold_matching(similarity: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Return a maximum-cardinality one-to-one match above ``threshold``."""
    matrix = np.asarray(similarity, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("similarity must be a two-dimensional matrix")
    n_generated, n_reference = matrix.shape
    candidates = [
        sorted(
            (j for j in range(n_reference) if matrix[i, j] >= threshold),
            key=lambda j: (-matrix[i, j], j),
        )
        for i in range(n_generated)
    ]
    reference_to_generated: dict[int, int] = {}

    def augment(generated_index: int, seen: set[int]) -> bool:
        for reference_index in candidates[generated_index]:
            if reference_index in seen:
                continue
            seen.add(reference_index)
            incumbent = reference_to_generated.get(reference_index)
            if incumbent is None or augment(incumbent, seen):
                reference_to_generated[reference_index] = generated_index
                return True
        return False

    for generated_index in sorted(range(n_generated), key=lambda i: (len(candidates[i]), i)):
        augment(generated_index, set())
    return sorted((g, r) for r, g in reference_to_generated.items())


def prf_from_similarity(
    similarity: np.ndarray,
    *,
    threshold: float,
    matching_policy: str = "independent_max",
) -> Dict[str, float | int]:
    """Calculate semantic precision, recall and F1 from raw similarities.

    ``independent_max`` is the registered many-to-one rule. ``one_to_one`` is
    a redundancy-sensitive maximum-cardinality threshold matching.
    """
    matrix = np.asarray(similarity, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("similarity must be a two-dimensional matrix")
    n_generated, n_reference = matrix.shape
    if n_generated == 0 or n_reference == 0:
        return {
            "n_generated": n_generated,
            "n_reference": n_reference,
            "n_generated_matched": 0,
            "n_reference_matched": 0,
            "n_matched_pairs": 0,
            "semantic_precision": 0.0,
            "semantic_recall": 0.0,
            "semantic_f1": 0.0,
        }

    if matching_policy == "independent_max":
        generated_hits = int(np.count_nonzero(matrix.max(axis=1) >= threshold))
        reference_hits = int(np.count_nonzero(matrix.max(axis=0) >= threshold))
        matched_pairs = -1
    elif matching_policy == "one_to_one":
        matched_pairs = len(maximum_threshold_matching(matrix, threshold))
        generated_hits = reference_hits = matched_pairs
    else:
        raise ValueError(f"unknown matching_policy: {matching_policy}")

    precision = generated_hits / n_generated
    recall = reference_hits / n_reference
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_generated": n_generated,
        "n_reference": n_reference,
        "n_generated_matched": generated_hits,
        "n_reference_matched": reference_hits,
        "n_matched_pairs": matched_pairs,
        "semantic_precision": precision,
        "semantic_recall": recall,
        "semantic_f1": f1,
    }


def semantic_prf(
    human_xml: Path,
    auto_xml: Path,
    *,
    model,
    threshold: float,
) -> Dict[str, Optional[float]]:
    """
    Compute semantic precision/recall/F1 between two FMs (by feature names, cosine similarity).
    - human_xml: ground-truth FM XML
    - auto_xml: generated FM XML
    - model: sentence-transformers model (encode, normalize_embeddings, convert_to_tensor)
    - threshold: cosine threshold to count a match
    """
    try:
        human_names = [h for h, _ in extract_nodes(human_xml)]
        auto_names = [a for a, _ in extract_nodes(auto_xml)]
        if not human_names or not auto_names:
            return {"semantic_precision": None, "semantic_recall": None, "semantic_f1": None}
        scored = prf_from_similarity(
            cosine_similarity_matrix(human_names, auto_names, model=model),
            threshold=threshold,
            matching_policy="independent_max",
        )
        return {
            "semantic_precision": round(float(scored["semantic_precision"]), 4),
            "semantic_recall": round(float(scored["semantic_recall"]), 4),
            "semantic_f1": round(float(scored["semantic_f1"]), 4),
        }
    except Exception:
        return {"semantic_precision": None, "semantic_recall": None, "semantic_f1": None}


def evaluate_semantic(
    fm_gen_path: Path,
    fm_ref_path: Path,
    *,
    encoder,
    tau_primary: float,
    tau_sweep: Sequence[float] = (0.3, 0.4, 0.5, 0.6),
    attested: Iterable[str] = (),
    organising: Iterable[str] = (),
    reach: Iterable[str] = (),
    rho: Optional[float] = None,
    attribution: Optional[Dict[str, set[str]]] = None,
) -> Dict[str, Any]:
    """Semantic evaluator with explicit metric envelopes.

    Returns ``{"metrics": {name: envelope}, "pairs": [dict...], "matrix_shape":
    (n_gen, n_ref), "gen_names": [...], "ref_names": [...]}``.

    ``pairs`` is the raw D15 payload: one row per (generated_occurrence,
    reference_occurrence, similarity) — no thresholding applied. τ sweeps are
    re-thresholded from this list downstream.

    ``recall_*`` denominators come from the D21 partition; ``recall_reach`` is
    the primary as required by the spec.

    ``ineligible`` metrics come back when either FM has zero feature nodes;
    ``missing_artifact`` when a file is absent; ``evaluator_error`` when the
    encoder or XML parser raises.
    """
    if not fm_gen_path.is_file():
        result = {k: _envelope(status="missing_artifact", reason=f"generated FM absent: {fm_gen_path}") for k in SEMANTIC_METRICS}
        return dict(metrics=result, pairs=[], matrix_shape=(0, 0), gen_names=[], ref_names=[])
    if not fm_ref_path.is_file():
        result = {k: _envelope(status="missing_artifact", reason=f"reference FM absent: {fm_ref_path}") for k in SEMANTIC_METRICS}
        return dict(metrics=result, pairs=[], matrix_shape=(0, 0), gen_names=[], ref_names=[])

    try:
        gen_nodes = extract_nodes(fm_gen_path)
        ref_nodes = extract_nodes(fm_ref_path)
    except Exception as exc:
        result = {k: _envelope(status="evaluator_error",
                               reason=f"{type(exc).__name__}: {exc}") for k in SEMANTIC_METRICS}
        return dict(metrics=result, pairs=[], matrix_shape=(0, 0), gen_names=[], ref_names=[])

    gen_names = [name for name, _ in gen_nodes]
    ref_names = [name for name, _ in ref_nodes]

    F_t_att = frozenset(attested)
    F_t_org = frozenset(organising)
    F_t = F_t_att | F_t_org
    reach_set = frozenset(reach)
    if F_t_att & F_t_org or not reach_set <= F_t:
        raise ValueError("Invalid semantic partitions: overlap or reach outside reference partition")
    computed_rho = len(reach_set) / len(F_t) if F_t else None
    if rho is not None and (computed_rho is None or
                            not math.isclose(rho, computed_rho, rel_tol=0, abs_tol=0.00005)):
        raise ValueError("rho must equal reach_size / F_t_size")

    partition_metrics = {
        "F_t_size": _envelope(len(F_t)),
        "F_t_attested_size": _envelope(len(F_t_att)),
        "F_t_organising_size": _envelope(len(F_t_org)),
        "reach_size": _envelope(len(reach_set)),
        "rho_T_C": _envelope(computed_rho,
                              status="ok" if F_t else "not_applicable",
                              reason="" if F_t else "Empty F_t; ρ undefined",
                              numerator=len(reach_set), denominator=len(F_t)),
    }

    if not gen_names or not ref_names:
        result = _ineligible("Empty feature set in generated or reference FM; scores undefined")
        result["n_generated"] = _envelope(len(gen_names))
        result["n_reference"] = _envelope(len(ref_names))
        result.update(partition_metrics)
        return dict(metrics=result, pairs=[], matrix_shape=(len(gen_names), len(ref_names)),
                    gen_names=gen_names, ref_names=ref_names)

    try:
        similarity = cosine_similarity_matrix(ref_names, gen_names, model=encoder)
        generated_self = cosine_similarity_matrix(gen_names, gen_names, model=encoder)
    except Exception as exc:
        result = {k: _envelope(status="evaluator_error",
                               reason=f"encoder failure: {type(exc).__name__}: {exc}") for k in SEMANTIC_METRICS}
        result["n_generated"] = _envelope(len(gen_names))
        result["n_reference"] = _envelope(len(ref_names))
        result.update(partition_metrics)
        return dict(metrics=result, pairs=[], matrix_shape=(len(gen_names), len(ref_names)),
                    gen_names=gen_names, ref_names=ref_names)

    matrix = np.asarray(similarity, dtype=float)
    from fame.evaluation.alignment import extract_indexed_nodes, evaluate_alignment
    try:
        indexed_generated = extract_indexed_nodes(fm_gen_path)
        indexed_reference = extract_indexed_nodes(fm_ref_path)
        if ([node["name"] for node in indexed_generated] != gen_names or
                [node["name"] for node in indexed_reference] != ref_names):
            raise ValueError("Feature occurrence order disagrees with semantic embedding order")
    except Exception as exc:
        result = {k: _envelope(status="evaluator_error",
            reason=f"feature occurrence extraction failure: {type(exc).__name__}: {exc}")
            for k in SEMANTIC_METRICS}
        result["n_generated"] = _envelope(len(gen_names))
        result["n_reference"] = _envelope(len(ref_names))
        result.update(partition_metrics)
        return dict(metrics=result, pairs=[], matrix_shape=matrix.shape,
                    gen_names=gen_names, ref_names=ref_names)
    pairs = [
        dict(
            gen_index=int(i), gen_name=gen_names[i],
            gen_parent_index=indexed_generated[i]["parent_index"],
            ref_index=int(j), ref_name=ref_names[j],
            ref_parent_index=indexed_reference[j]["parent_index"],
            similarity=float(matrix[i, j]),
        )
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
    ]

    ref_max = matrix.max(axis=0) if matrix.size else np.array([])
    matched_ref_names = {ref_names[j] for j in range(len(ref_names))
                         if ref_max.size and ref_max[j] >= tau_primary}
    gen_hits_bool = matrix.max(axis=1) >= tau_primary if matrix.size else np.array([], dtype=bool)
    n_generated_matched = int(gen_hits_bool.sum())

    if len(gen_names) > 0:
        precision = n_generated_matched / len(gen_names)
        precision_env = _envelope(precision, numerator=n_generated_matched, denominator=len(gen_names))
    else:
        precision_env = _envelope(status="not_applicable", reason="No generated features")

    recall_total = len(matched_ref_names) / len(ref_names)
    recall_total_env = _envelope(recall_total, numerator=len(matched_ref_names), denominator=len(ref_names))

    if precision + recall_total > 0:
        f1 = 2 * precision * recall_total / (precision + recall_total)
        f1_env = _envelope(f1)
    else:
        f1_env = _envelope(0.0, reason="Both precision and recall are zero")

    def _partition_recall(target: frozenset[str], label: str) -> dict:
        if not target:
            return _envelope(status="not_applicable",
                             reason=f"Empty {label} partition; recall undefined")
        hits = matched_ref_names & target
        return _envelope(len(hits) / len(target),
                         numerator=len(hits), denominator=len(target))

    ref_hit_counts = ((matrix >= tau_primary).sum(axis=0)).tolist() if matrix.size else []
    duplicate_hits = int(sum(1 for c in ref_hit_counts if c > 1))

    result: Dict[str, dict] = {
        "n_generated": _envelope(len(gen_names)),
        "n_reference": _envelope(len(ref_names)),
        "semantic_precision": precision_env,
        "semantic_recall_total": recall_total_env,
        "semantic_f1_total": f1_env,
        "recall_reach": _partition_recall(reach_set, "reach"),
        "recall_attested": _partition_recall(F_t_att, "attested"),
        "recall_organising": _partition_recall(F_t_org, "organising"),
        "matched_reference_ids": _envelope(sorted(matched_ref_names)),
        "duplicate_generated_hits": _envelope(duplicate_hits,
            reason="reference nodes receiving more than one above-τ generated match; report alongside one-to-one sensitivity"),
    }
    result.update(partition_metrics)

    alignment_keys = (
        "n_parent_evaluable", "n_parent_correct", "parent_match_rate",
        "n_exact_parent_evaluable", "n_exact_parent_correct", "exact_name_parent_match_rate",
        "exact_duplicate_surplus", "near_duplicate_pairs_tau_0_8", "near_duplicate_pairs_tau_0_9",
        "n_citations", "n_attribution_applicable", "n_attribution_agreeing",
        "attribution_agreement_rate", "n_citations_unmapped",
        "n_citations_reference_unannotated",
    )
    try:
        aligned = evaluate_alignment(
            matrix, indexed_generated, indexed_reference,
            tau=tau_primary, generated_self_similarity=generated_self,
            attribution=attribution,
        )
        for key in alignment_keys:
            value = aligned[key]
            result[key] = (_envelope(value) if value is not None else
                           _envelope(status="not_applicable", reason="No eligible matched relationship or annotated citation"))
        for key, count_key in (("parent_match_rate", "n_parent_evaluable"),
                               ("exact_name_parent_match_rate", "n_exact_parent_evaluable"),
                               ("attribution_agreement_rate", "n_attribution_applicable")):
            if result[key]["status"] == "ok":
                result[key]["denominator"] = aligned[count_key]
                numerator_key = {"parent_match_rate": "n_parent_correct",
                                 "exact_name_parent_match_rate": "n_exact_parent_correct",
                                 "attribution_agreement_rate": "n_attribution_agreeing"}[key]
                result[key]["numerator"] = aligned[numerator_key]
    except Exception as exc:
        for key in alignment_keys:
            result[key] = _envelope(status="evaluator_error",
                reason=f"alignment failure: {type(exc).__name__}: {exc}")

    tau_sweep_view: Dict[str, Dict[float, float]] = {}
    for tau in tau_sweep:
        ref_max_t = ref_max
        gen_max_t = matrix.max(axis=1) if matrix.size else np.array([])
        ref_hits_t = int((ref_max_t >= tau).sum()) if ref_max_t.size else 0
        gen_hits_t = int((gen_max_t >= tau).sum()) if gen_max_t.size else 0
        p_t = gen_hits_t / len(gen_names)
        r_t = ref_hits_t / len(ref_names)
        f1_t = (2 * p_t * r_t / (p_t + r_t)) if (p_t + r_t) else 0.0
        tau_sweep_view.setdefault("precision", {})[float(tau)] = p_t
        tau_sweep_view.setdefault("recall_total", {})[float(tau)] = r_t
        tau_sweep_view.setdefault("f1_total", {})[float(tau)] = f1_t

    return dict(metrics=result, pairs=pairs, matrix_shape=matrix.shape,
                gen_names=gen_names, ref_names=ref_names,
                tau_primary=tau_primary, tau_sweep_view=tau_sweep_view)


def feature_diff_stats(human_xml: Path, auto_xml: Path) -> Dict[str, Optional[float]]:
    """
    Compute extra/missing feature counts and ratios (name-based).
    - extra = generated but not in GT
    - missing = GT not generated
    Ratios are over generated and GT counts respectively.
    """
    try:
        human = {h for h, _ in extract_nodes(human_xml)}
        auto = {a for a, _ in extract_nodes(auto_xml)}
        extra = auto - human
        missing = human - auto
        auto_n = len(auto)
        human_n = len(human)
        return {
            "extra_feature_count": len(extra),
            "missing_feature_count": len(missing),
            "extra_feature_ratio": round(len(extra) / auto_n, 4) if auto_n else None,
            "missing_feature_ratio": round(len(missing) / human_n, 4) if human_n else None,
        }
    except Exception:
        return {
            "extra_feature_count": None,
            "missing_feature_count": None,
            "extra_feature_ratio": None,
            "missing_feature_ratio": None,
        }
