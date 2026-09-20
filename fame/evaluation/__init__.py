"""Evaluation utilities for Feature Models.

Current modules:
 - coverage: semantic recall-style coverage of generated vs ground-truth FMs.

Designed to be extended (structure, constraints, traceability, etc.).
"""

from .coverage import CoverageEvaluator, CoverageConfig, coverage_score
from .wellformed import validate_feature_model, WellFormedResult
from .duration import start_timer, elapsed_seconds
from .feature_list import extract_feature_list, FeatureRecord
from .constraints import extract_constraints, ConstraintRecord
from .run_metadata import RunMetadata, write_run_metadata, default_timestamp
from .semantic import semantic_prf, feature_diff_stats
from .structure import edge_jaccard_vs_gt, parent_match_rate
from .quality_sat import analyze_sat_quality, SATQuality
from .proxy_evidence import score_evidence
from .proxy_consensus import extract_signature, score_consensus
from .proxy_selector import ProxyRankConfig, build_proxy_rows, rank_proxy_fms, check_admissibility

__all__ = [
    'CoverageEvaluator',
    'CoverageConfig',
    'coverage_score',
    'validate_feature_model',
    'WellFormedResult',
    'start_timer',
    'elapsed_seconds',
    'extract_feature_list',
    'FeatureRecord',
    'extract_constraints',
    'ConstraintRecord',
    'RunMetadata',
    'write_run_metadata',
    'default_timestamp',
    'semantic_prf',
    'feature_diff_stats',
    'edge_jaccard_vs_gt',
    'parent_match_rate',
    'analyze_sat_quality',
    'SATQuality',
    'score_evidence',
    'extract_signature',
    'score_consensus',
    'ProxyRankConfig',
    'build_proxy_rows',
    'rank_proxy_fms',
    'check_admissibility',
]
