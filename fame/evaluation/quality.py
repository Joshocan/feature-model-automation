from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Set, Union

from .depth import extract_depths


from typing import Optional

FeatureParentMap = Dict[str, Optional[str]]


def _extract_features_and_parents(xml_path: Union[str, Path]) -> FeatureParentMap:
    """
    Returns mapping feature_name -> direct_parent_name (or None for root).
    Groups (and/or/alt) are traversed but not counted as features.
    """
    tree = ET.parse(Path(xml_path).expanduser())
    root = tree.getroot()
    struct = root.find("struct")
    if struct is None:
        raise ValueError("Invalid FeatureIDE XML: <struct> not found")

    parents: FeatureParentMap = {}

    def walk(node: ET.Element, parent_name: Optional[str]):
        tag = node.tag
        name = node.attrib.get("name")
        is_feature = tag == "feature"

        current_parent = parent_name
        if is_feature and name:
            parents[name] = parent_name
            current_parent = name

        for child in node:
            walk(child, current_parent)

    for child in struct:
        walk(child, None)
    return parents


def _extract_abstract_concrete_counts(xml_path: Union[str, Path]) -> tuple[int, int]:
    """
    Counts abstract vs concrete features.
    FeatureIDE marks abstract features with abstract="true" on <feature>.
    Groups are ignored.
    """
    tree = ET.parse(Path(xml_path).expanduser())
    root = tree.getroot()
    struct = root.find("struct")
    if struct is None:
        raise ValueError("Invalid FeatureIDE XML: <struct> not found")

    abstract = 0
    concrete = 0
    for el in struct.iter():
        if el.tag != "feature":
            continue
        if el.attrib.get("abstract", "").lower() == "true":
            abstract += 1
        else:
            concrete += 1
    return abstract, concrete


def _count_groups(xml_path: Union[str, Path]) -> tuple[int, int]:
    tree = ET.parse(Path(xml_path).expanduser())
    root = tree.getroot()
    struct = root.find("struct")
    if struct is None:
        return 0, 0
    or_cnt = 0
    alt_cnt = 0
    for el in struct.iter():
        if el.tag == "or":
            or_cnt += 1
        elif el.tag == "alt":
            alt_cnt += 1
    return or_cnt, alt_cnt


def _extract_constraint_feature_refs(xml_path: Union[str, Path]) -> Set[str]:
    tree = ET.parse(Path(xml_path).expanduser())
    root = tree.getroot()
    constraints = root.find("constraints")
    if constraints is None:
        return set()

    names: Set[str] = set()
    for el in constraints.iter():
        for key in ("name", "feature", "var"):
            val = el.attrib.get(key)
            if val:
                names.add(val)
    return names


def _count_constraints(xml_path: Union[str, Path]) -> int:
    tree = ET.parse(Path(xml_path).expanduser())
    root = tree.getroot()
    constraints = root.find("constraints")
    if constraints is None:
        return 0
    # FeatureIDE typically wraps each formula in <rule>
    rule_tags = list(constraints.findall(".//rule"))
    return len(rule_tags) if rule_tags else len(list(constraints))


@dataclass
class QualityMetrics:
    feature_count: int
    abstract_count: int
    concrete_count: int
    abstract_concrete_ratio: float
    mandatory_ratio: float
    optional_ratio: float
    max_depth: int
    avg_depth: float
    or_group_count: int
    alt_group_count: int
    group_density: float
    constraint_count: int
    constraint_feature_ratio: float
    orphan_features: List[str]
    duplicate_features: List[str]
    broken_constraint_refs: List[str]


def analyze_quality(xml_path: Union[str, Path]) -> QualityMetrics:
    xml_path = Path(xml_path).expanduser()

    # Feature set and parent map
    parents = _extract_features_and_parents(xml_path)
    feature_names = set(parents.keys())
    feature_count = len(feature_names)

    # Duplicates
    dupes = [name for name, cnt in Counter(parents.keys()).items() if cnt > 1]

    # Orphans (parent name missing from feature set)
    orphans = sorted({f for f, p in parents.items() if p is not None and p not in feature_names})

    # Depth metrics
    depths = extract_depths(xml_path)
    depth_vals = list(depths.values())
    max_depth = max(depth_vals) if depth_vals else 0
    avg_depth = mean(depth_vals) if depth_vals else 0.0

    # Abstract / concrete
    abstract_cnt, concrete_cnt = _extract_abstract_concrete_counts(xml_path)
    ratio = abstract_cnt / concrete_cnt if concrete_cnt > 0 else float("inf") if abstract_cnt > 0 else 0.0

    # Mandatory / optional ratios
    mandatory_edges = sum(1 for _, p in parents.items() if p is not None)  # every child edge counts
    # Optional approximation: treat features tagged optional="true" as optional, else mandatory
    # We also expose a ratio of mandatory edges to total edges (features with parents).
    opt_count = 0
    tree = ET.parse(xml_path)
    root = tree.getroot()
    struct = root.find("struct")
    if struct is not None:
        for el in struct.iter():
            if el.tag == "feature" and el.attrib.get("optional", "").lower() == "true":
                opt_count += 1
    mand_ratio = (mandatory_edges - opt_count) / mandatory_edges if mandatory_edges > 0 else 0.0
    opt_ratio = opt_count / mandatory_edges if mandatory_edges > 0 else 0.0

    # Groups
    or_cnt, alt_cnt = _count_groups(xml_path)
    internal_nodes = max(1, len(parents))  # avoid div0; crude proxy
    group_density = (or_cnt + alt_cnt) / internal_nodes

    # Constraints
    constraint_cnt = _count_constraints(xml_path)
    constraint_ratio = constraint_cnt / feature_count if feature_count > 0 else 0.0
    constraint_refs = _extract_constraint_feature_refs(xml_path)
    broken_refs = sorted(list(constraint_refs - feature_names))

    return QualityMetrics(
        feature_count=feature_count,
        abstract_count=abstract_cnt,
        concrete_count=concrete_cnt,
        abstract_concrete_ratio=round(ratio, 4) if ratio != float("inf") else float("inf"),
        mandatory_ratio=round(mand_ratio, 4),
        optional_ratio=round(opt_ratio, 4),
        max_depth=max_depth,
        avg_depth=round(avg_depth, 4),
        or_group_count=or_cnt,
        alt_group_count=alt_cnt,
        group_density=round(group_density, 4),
        constraint_count=constraint_cnt,
        constraint_feature_ratio=round(constraint_ratio, 4),
        orphan_features=orphans,
        duplicate_features=dupes,
        broken_constraint_refs=broken_refs,
    )
