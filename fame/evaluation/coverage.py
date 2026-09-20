from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    _HAS_ST = False


Node = Tuple[str, Optional[str]]  # (name, parent_name)


def extract_nodes(xml_file: Union[Path, str]) -> List[Node]:
    """
    Extract (node_name, direct_parent_name) for ALL nodes from FeatureIDE-style XML.
    Includes feature/group nodes; skips missing names.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    struct = root.find("struct")
    if struct is None:
        raise ValueError("Invalid FeatureIDE XML: <struct> not found")

    nodes: List[Node] = []

    def walk(node, parent_name=None):
        name = node.attrib.get("name")

        if node.tag in {"and", "or", "alt", "feature"} and name:
            nodes.append((name, parent_name))
            current_parent = name
        else:
            current_parent = parent_name

        for child in node:
            walk(child, current_parent)

    for child in struct:
        walk(child, None)

    return nodes


@dataclass
class CoverageConfig:
    model_name: str
    similarity_threshold: float
    top_k: int
    feature_weight: float
    parent_weight: float


class CoverageEvaluator:
    """Semantic coverage (recall) of AUTO FM vs HUMAN FM."""

    def __init__(self, cfg: CoverageConfig):
        self.cfg = cfg
        if not _HAS_ST:
            raise ImportError("sentence_transformers is required for coverage scoring but is not installed.")
        self.model = SentenceTransformer(self.cfg.model_name)

    def _encode(self, texts: Iterable[str]):
        return self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_tensor=True
        )

    def score(self, human_xml: Union[Path, str], auto_xml: Union[Path, str], verbose: bool = True) -> float:
        human_nodes = extract_nodes(human_xml)
        auto_nodes = extract_nodes(auto_xml)

        if not human_nodes:
            return 0.0

        human_names = [h for h, _ in human_nodes]
        auto_names = [a for a, _ in auto_nodes]

        human_emb = self._encode(human_names)
        auto_emb = self._encode(auto_names)

        # Parent embeddings (only for nodes that have a parent)
        human_parent_emb = {
            hp: self._encode([hp])[0] for _, hp in human_nodes if hp
        }
        auto_parent_emb = {
            ap: self._encode([ap])[0] for _, ap in auto_nodes if ap
        }

        total_coverage = 0.0

        if verbose:
            print("\n=== Coverage Matches (Human → Auto) ===\n")

        for i, (hx, hp) in enumerate(human_nodes):
            similarities = []
            for j, (ax, ap) in enumerate(auto_nodes):
                s_node = util.cos_sim(human_emb[i], auto_emb[j]).item()

                if hp and ap and hp in human_parent_emb and ap in auto_parent_emb:
                    s_parent = util.cos_sim(
                        human_parent_emb[hp], auto_parent_emb[ap]
                    ).item()
                else:
                    s_parent = 0.0

                score = self.cfg.feature_weight * s_node + self.cfg.parent_weight * s_parent
                similarities.append((score, ax, ap, s_node, s_parent))

            similarities.sort(reverse=True, key=lambda x: x[0])
            top_matches = similarities[: self.cfg.top_k]
            valid_scores = [s for s, *_ in top_matches if s >= self.cfg.similarity_threshold]

            if valid_scores:
                coverage = 1 - np.prod([1 - s for s in valid_scores])
                total_coverage += coverage

                if verbose:
                    print(f"HUMAN NODE : {hx}")
                    for s, ax, ap, sn, sp in top_matches:
                        if s >= self.cfg.similarity_threshold:
                            print(
                                f"  ↳ AUTO NODE: {ax} | score={round(s,3)} "
                                f"(node={round(sn,3)}, parent={round(sp,3)})"
                            )
                    print("-" * 60)
            else:
                if verbose:
                    print(f"HUMAN NODE : {hx}")
                    print("  ↳ NO ADEQUATE COVERAGE")
                    print("-" * 60)

        final_score = (total_coverage / len(human_nodes)) * 100
        if verbose:
            print(f"\nCoverage Score (Recall): {round(final_score, 2)}/100\n")
        return round(final_score, 2)


def coverage_score(human_xml: Union[Path, str], auto_xml: Union[Path, str], **kwargs) -> float:
    return CoverageEvaluator(CoverageConfig(**kwargs)).score(human_xml, auto_xml, verbose=False)
