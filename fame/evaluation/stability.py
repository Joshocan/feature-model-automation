from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from .depth import extract_depths
from .feature_list import extract_feature_list
from .quality import _extract_features_and_parents, _extract_constraint_feature_refs

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore

    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    _HAS_ST = False


FeatureSet = Set[str]
EdgeSet = Set[Tuple[str, str]]


def _edge_set(xml_path: Union[str, Path]) -> EdgeSet:
    parents = _extract_features_and_parents(xml_path)
    return {(child, parent) for child, parent in parents.items() if parent is not None}


def _constraint_strings(xml_path: Union[str, Path]) -> List[str]:
    # Reuse constraint refs only to get text; fall back to empty if none.
    refs = _extract_constraint_feature_refs(xml_path)
    if refs:
        return sorted(refs)
    return []


def _centroid(texts: List[str], model: SentenceTransformer) -> Optional[np.ndarray]:
    if not texts:
        return None
    emb = model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
    return emb.mean(dim=0)


def _pairwise_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _mean_pairwise_cosine(vectors: List[np.ndarray]) -> float:
    pairs = []
    for a, b in itertools.combinations(vectors, 2):
        pairs.append(float(util.cos_sim(a, b)))
    return _pairwise_mean(pairs)


def _mean_pairwise_jaccard(sets: List[Set]) -> float:
    pairs = []
    for a, b in itertools.combinations(sets, 2):
        if not a and not b:
            pairs.append(1.0)
            continue
        inter = len(a & b)
        uni = len(a | b)
        pairs.append(inter / uni if uni else 0.0)
    return _pairwise_mean(pairs)


@dataclass
class StabilityMetrics:
    n_runs: int
    std_feature_count: float
    std_max_depth: float
    std_avg_depth: float
    feature_jaccard_mean: float
    edge_jaccard_mean: float
    feature_cosine_mean: Optional[float]
    constraint_cosine_mean: Optional[float]
    per_run_feature_counts: List[int]
    per_run_max_depths: List[int]
    per_run_avg_depths: List[float]


def compute_stability(
    fm_paths: List[Union[str, Path]],
    *,
    embed_model: str = "all-MiniLM-L6-v2",
    include_constraints: bool = False,
) -> StabilityMetrics:
    fm_paths = [Path(p).expanduser() for p in fm_paths]
    if len(fm_paths) < 2:
        # Not enough runs to compute pairwise; return zeros.
        return StabilityMetrics(
            n_runs=len(fm_paths),
            std_feature_count=0.0,
            std_max_depth=0.0,
            std_avg_depth=0.0,
            feature_jaccard_mean=0.0,
            edge_jaccard_mean=0.0,
            feature_cosine_mean=None,
            constraint_cosine_mean=None,
            per_run_feature_counts=[],
            per_run_max_depths=[],
            per_run_avg_depths=[],
        )

    # Load embedding model if cosine metrics requested
    model: Optional[SentenceTransformer] = None
    if _HAS_ST:
        model = SentenceTransformer(embed_model)
    else:
        if include_constraints:
            raise ImportError("sentence_transformers is required for cosine metrics.")

    feature_sets: List[FeatureSet] = []
    edge_sets: List[EdgeSet] = []
    feature_centroids: List[np.ndarray] = []
    constraint_centroids: List[np.ndarray] = []
    feature_counts: List[int] = []
    max_depths: List[int] = []
    avg_depths: List[float] = []

    for fm in fm_paths:
        feat_records = extract_feature_list(fm)
        feats = {fr.feature_name for fr in feat_records}
        feature_sets.append(feats)
        feature_counts.append(len(feats))

        edges = _edge_set(fm)
        edge_sets.append(edges)

        depths = extract_depths(fm)
        depth_vals = list(depths.values())
        max_depths.append(max(depth_vals) if depth_vals else 0)
        avg_depths.append(sum(depth_vals) / len(depth_vals) if depth_vals else 0.0)

        if model:
            c = _centroid(sorted(feats), model)
            if c is not None:
                feature_centroids.append(c)
            if include_constraints:
                cs = _centroid(_constraint_strings(fm), model)
                if cs is not None:
                    constraint_centroids.append(cs)

    std_feat = float(np.std(feature_counts, ddof=0)) if feature_counts else 0.0
    std_max = float(np.std(max_depths, ddof=0)) if max_depths else 0.0
    std_avg = float(np.std(avg_depths, ddof=0)) if avg_depths else 0.0

    feat_j = _mean_pairwise_jaccard(feature_sets)
    edge_j = _mean_pairwise_jaccard(edge_sets)

    feat_cos = None
    if feature_centroids and len(feature_centroids) > 1:
        feat_cos = _mean_pairwise_cosine(feature_centroids)

    cons_cos = None
    if include_constraints and constraint_centroids and len(constraint_centroids) > 1:
        cons_cos = _mean_pairwise_cosine(constraint_centroids)

    return StabilityMetrics(
        n_runs=len(fm_paths),
        std_feature_count=round(std_feat, 4),
        std_max_depth=round(std_max, 4),
        std_avg_depth=round(std_avg, 4),
        feature_jaccard_mean=round(feat_j, 4),
        edge_jaccard_mean=round(edge_j, 4),
        feature_cosine_mean=round(feat_cos, 4) if feat_cos is not None else None,
        constraint_cosine_mean=round(cons_cos, 4) if cons_cos is not None else None,
        per_run_feature_counts=feature_counts,
        per_run_max_depths=max_depths,
        per_run_avg_depths=[round(v, 4) for v in avg_depths],
    )


def iteration_jaccard(fm_paths: List[Union[str, Path]]) -> Dict[str, List[float]]:
    """
    Jaccard similarity between consecutive iterations (features and edges).
    """
    fm_paths = [Path(p).expanduser() for p in fm_paths]
    feat_sets = [set(extract_feature_list(p)) for p in fm_paths]
    edge_sets = [_edge_set(p) for p in fm_paths]

    feat_seq = []
    edge_seq = []
    for i in range(len(fm_paths) - 1):
        a, b = feat_sets[i], feat_sets[i + 1]
        ea, eb = edge_sets[i], edge_sets[i + 1]
        feat_seq.append(_mean_pairwise_jaccard([a, b]))
        edge_seq.append(_mean_pairwise_jaccard([ea, eb]))
    return {"feature_jaccard_seq": [round(x, 4) for x in feat_seq], "edge_jaccard_seq": [round(x, 4) for x in edge_seq]}


def iteration_growth(fm_paths: List[Union[str, Path]]) -> Dict[str, List[int]]:
    """
    Feature growth per iteration: delta, new, dropped counts between consecutive FMs.
    """
    fm_paths = [Path(p).expanduser() for p in fm_paths]
    feat_sets = [set(extract_feature_list(p)) for p in fm_paths]
    delta = []
    new_counts = []
    drop_counts = []
    for i in range(len(fm_paths) - 1):
        a, b = feat_sets[i], feat_sets[i + 1]
        delta.append(len(b) - len(a))
        new_counts.append(len(b - a))
        drop_counts.append(len(a - b))
    return {
        "delta_features": delta,
        "new_features": new_counts,
        "dropped_features": drop_counts,
    }


def iteration_depth_change(fm_paths: List[Union[str, Path]]) -> Dict[str, List[float]]:
    """
    Depth changes between consecutive iterations (max and avg).
    """
    fm_paths = [Path(p).expanduser() for p in fm_paths]
    max_depths = [max(extract_depths(p).values()) if extract_depths(p) else 0 for p in fm_paths]
    avg_depths = [
        (sum(extract_depths(p).values()) / len(extract_depths(p))) if extract_depths(p) else 0.0 for p in fm_paths
    ]
    delta_max = []
    delta_avg = []
    for i in range(len(fm_paths) - 1):
        delta_max.append(max_depths[i + 1] - max_depths[i])
        delta_avg.append(avg_depths[i + 1] - avg_depths[i])
    return {
        "delta_max_depth": [round(x, 4) for x in delta_max],
        "delta_avg_depth": [round(x, 4) for x in delta_avg],
    }
