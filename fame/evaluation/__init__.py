"""Evaluation utilities for feature models.

Symbolic + semantic + structural primitives consumed by the campaign analysis stages.
Proxy-selector modules were removed with the SS/IS pipeline retirement (arm dropped from the
new design).
"""

from .constraints import extract_constraints, ConstraintRecord
from .coverage import CoverageEvaluator, CoverageConfig, coverage_score
from .duration import start_timer, elapsed_seconds
from .feature_list import extract_feature_list, FeatureRecord
from .quality_sat import analyze_sat_quality, SATQuality
from .run_metadata import RunMetadata, write_run_metadata, default_timestamp
from .semantic import (
    cosine_similarity_matrix,
    feature_diff_stats,
    maximum_threshold_matching,
    prf_from_similarity,
    semantic_prf,
)
from .structure import edge_jaccard_vs_gt, parent_match_rate
from .wellformed import validate_feature_model, WellFormedResult

__all__ = [
    "CoverageEvaluator",
    "CoverageConfig",
    "ConstraintRecord",
    "FeatureRecord",
    "RunMetadata",
    "SATQuality",
    "WellFormedResult",
    "analyze_sat_quality",
    "coverage_score",
    "cosine_similarity_matrix",
    "default_timestamp",
    "edge_jaccard_vs_gt",
    "elapsed_seconds",
    "extract_constraints",
    "extract_feature_list",
    "feature_diff_stats",
    "maximum_threshold_matching",
    "parent_match_rate",
    "prf_from_similarity",
    "semantic_prf",
    "start_timer",
    "validate_feature_model",
    "write_run_metadata",
]
