from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

from .coverage import extract_nodes


def _edges(xml_path: Path) -> Set[Tuple[str, Optional[str]]]:
    return {(name, parent) for name, parent in extract_nodes(xml_path)}


def edge_jaccard_vs_gt(human_xml: Path, auto_xml: Path) -> Optional[float]:
    """Jaccard of (child,parent) pairs; parents may be None. None if both empty."""
    try:
        h = _edges(human_xml)
        a = _edges(auto_xml)
        if not h and not a:
            return None
        inter = len(h & a)
        union = len(h | a)
        return round(inter / union, 4) if union else None
    except Exception:
        return None


def parent_match_rate(human_xml: Path, auto_xml: Path) -> Optional[float]:
    """
    Percent of shared features whose parent matches GT (None allowed).
    Only features present in both models are considered.
    """
    try:
        h_edges = dict(extract_nodes(human_xml))
        a_edges = dict(extract_nodes(auto_xml))
        shared = set(h_edges.keys()) & set(a_edges.keys())
        if not shared:
            return None
        matches = sum(1 for f in shared if h_edges.get(f) == a_edges.get(f))
        return round(matches / len(shared), 4)
    except Exception:
        return None
