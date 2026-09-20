from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import xml.etree.ElementTree as ET

TRACE_RE = re.compile(r"Trace:\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)


@dataclass
class FeatureRecord:
    feature_id: str
    feature_name: str
    feature_type: str  # abstract | concrete
    hierarchy_level: int
    description: str
    evidence_refs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "hierarchy_level": self.hierarchy_level,
            "description": self.description,
            "evidence_refs": self.evidence_refs,
        }


def _description_text(node: ET.Element) -> str:
    desc = node.find("description")
    if desc is None or desc.text is None:
        return ""
    return desc.text.strip()


def _trace_refs(text: str) -> List[str]:
    m = TRACE_RE.search(text or "")
    if not m:
        return []
    inside = m.group(1)
    parts = [p.strip() for p in inside.split(";") if p.strip()]
    return parts


def _is_abstract(node: ET.Element, has_child_feature: bool) -> bool:
    abstract_attr = node.attrib.get("abstract", "false").lower() == "true"
    if abstract_attr:
        return True
    return has_child_feature


def extract_feature_list(xml_path: Path | str) -> List[FeatureRecord]:
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    struct = root.find("struct")
    if struct is None:
        raise ValueError("<struct> not found in feature model")

    records: List[FeatureRecord] = []
    counter = 0

    def walk(node: ET.Element, depth: int):
        nonlocal counter

        name = node.attrib.get("name")
        # Determine if node has feature children
        child_features = []
        for ch in node:
            if ch.tag in {"feature", "and", "or", "alt"}:
                child_features.append(ch)

        has_child_feature = len(child_features) > 0

        if name:
            counter += 1
            desc = _description_text(node)
            feature_type = "abstract" if _is_abstract(node, has_child_feature) else "concrete"
            refs = _trace_refs(desc)
            records.append(
                FeatureRecord(
                    feature_id=f"f{counter}",
                    feature_name=name,
                    feature_type=feature_type,
                    hierarchy_level=depth,
                    description=desc,
                    evidence_refs=refs,
                )
            )

        # Recurse into children; for groups (and/or/alt) we keep same depth if they don't have name
        next_depth = depth + 1 if name else depth
        for ch in node:
            if ch.tag in {"feature", "and", "or", "alt"}:
                walk(ch, next_depth)

    for child in struct:
        walk(child, depth=0)

    return records

