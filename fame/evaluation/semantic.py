from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .coverage import extract_nodes, util


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
        human_nodes = extract_nodes(human_xml)
        auto_nodes = extract_nodes(auto_xml)
        human_names = [h for h, _ in human_nodes]
        auto_names = [a for a, _ in auto_nodes]
        if not human_names or not auto_names:
            return {"semantic_precision": None, "semantic_recall": None, "semantic_f1": None}

        human_emb = model.encode(human_names, normalize_embeddings=True, convert_to_tensor=True)
        auto_emb = model.encode(auto_names, normalize_embeddings=True, convert_to_tensor=True)
        sim = util.cos_sim(auto_emb, human_emb)  # auto x human

        auto_max = sim.max(dim=1).values
        prec_matches = (auto_max >= threshold).sum().item()
        precision = prec_matches / len(auto_names) if auto_names else None

        human_max = sim.max(dim=0).values
        rec_matches = (human_max >= threshold).sum().item()
        recall = rec_matches / len(human_names) if human_names else None

        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0

        return {
            "semantic_precision": round(precision, 4) if precision is not None else None,
            "semantic_recall": round(recall, 4) if recall is not None else None,
            "semantic_f1": round(f1, 4) if f1 is not None else None,
        }
    except Exception:
        return {"semantic_precision": None, "semantic_recall": None, "semantic_f1": None}


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
