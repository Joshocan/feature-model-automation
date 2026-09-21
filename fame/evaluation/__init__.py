"""Evaluation utilities for feature models.

Symbolic + semantic + structural primitives consumed by Phase 9 analysis.
Proxy-selector modules were removed in Phase 1 Pass B (arm dropped from the
new design).
"""

from .constraints import extract_constraints, ConstraintRecord
from .coverage import CoverageEvaluator, CoverageConfig, coverage_score
from .duration import start_timer, elapsed_seconds
from .feature_list import extract_feature_list, FeatureRecord
from .quality_sat import analyze_sat_quality, SATQuality
from .run_metadata import RunMetadata, write_run_metadata, default_timestamp
from .semantic import semantic_prf, feature_diff_stats
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
    "default_timestamp",
    "edge_jaccard_vs_gt",
    "elapsed_seconds",
    "extract_constraints",
    "extract_feature_list",
    "feature_diff_stats",
    "parent_match_rate",
    "semantic_prf",
    "start_timer",
    "validate_feature_model",
    "write_run_metadata",
]
