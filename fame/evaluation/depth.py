from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple, Union, Optional


NodeDepths = Dict[str, int]


def _walk_depth(node, current_depth: int, out: NodeDepths) -> None:
    name = node.attrib.get("name")
    if name:
        out[name] = current_depth
        next_depth = current_depth + 1
    else:
        next_depth = current_depth

    for child in node:
        # FeatureIDE mixes feature/group tags; we still descend depth-wise.
        _walk_depth(child, next_depth, out)


def extract_depths(xml_file: Union[str, Path]) -> NodeDepths:
    """
    Return a mapping of feature name -> depth (root = 0) from a FeatureIDE XML file.
    Depth increases by 1 per edge from the root in the <struct> tree.
    """
    xml_path = Path(xml_file).expanduser()
    tree = ET.parse(xml_path)
    root = tree.getroot()

    struct = root.find("struct")
    if struct is None:
        raise ValueError("Invalid FeatureIDE XML: <struct> not found")

    depths: NodeDepths = {}
    for child in struct:
        _walk_depth(child, 0, depths)
    return depths


@dataclass
class DepthMetrics:
    mean_abs_error: float
    median_abs_error: float
    max_abs_error: int
    exact_match_rate: float
    compared_features: int


def depth_errors(
    gt_xml: Union[str, Path],
    auto_xml: Union[str, Path],
    *,
    feature_filter: Optional[Iterable[str]] = None,
) -> List[Tuple[str, int, int, int]]:
    """
    Compute per-feature depth errors for features present in BOTH GT and auto
    (exact name match). Returns list of tuples:
        (feature_name, gt_depth, auto_depth, abs_error)
    If feature_filter is provided, only those GT feature names are considered.
    """
    gt_depths = extract_depths(gt_xml)
    auto_depths = extract_depths(auto_xml)

    names = gt_depths.keys() if feature_filter is None else feature_filter

    results: List[Tuple[str, int, int, int]] = []
    for name in names:
        if name in auto_depths:
            err = abs(gt_depths[name] - auto_depths[name])
            results.append((name, gt_depths[name], auto_depths[name], err))
    return results


def depth_metrics(
    gt_xml: Union[str, Path],
    auto_xml: Union[str, Path],
    *,
    feature_filter: Optional[Iterable[str]] = None,
) -> DepthMetrics:
    """
    Aggregate depth error statistics over exact-name overlaps:
      - mean_abs_error
      - median_abs_error
      - max_abs_error
      - exact_match_rate (fraction with zero error)
      - compared_features (count)
    """
    rows = depth_errors(gt_xml, auto_xml, feature_filter=feature_filter)
    if not rows:
        return DepthMetrics(mean_abs_error=0.0, median_abs_error=0.0, max_abs_error=0, exact_match_rate=0.0, compared_features=0)

    errs = [r[3] for r in rows]
    mean_err = mean(errs)
    median_err = median(errs)
    max_err = max(errs)
    exact_rate = sum(1 for e in errs if e == 0) / len(errs)

    return DepthMetrics(
        mean_abs_error=round(mean_err, 4),
        median_abs_error=round(median_err, 4),
        max_abs_error=max_err,
        exact_match_rate=round(exact_rate, 4),
        compared_features=len(errs),
    )


# Convenience single-model depth stats (no GT)
def max_depth(xml_path: Union[str, Path]) -> int:
    """Return the maximum depth (root = 0) in a single FM XML."""
    depths = extract_depths(xml_path)
    return max(depths.values()) if depths else 0


def avg_depth(xml_path: Union[str, Path]) -> float:
    """Return the average depth (root = 0) in a single FM XML."""
    depths = extract_depths(xml_path)
    if not depths:
        return 0.0
    return sum(depths.values()) / len(depths)
